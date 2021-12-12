function [C_t,C]=MMTSC(Xt,Mask,option)

n_t=size(Xt);
N=length(n_t);
L=option.ndata;
mu=option.mu;
beta=option.beta;
lambda=option.lambda;

mode=option.l;
d=option.d;

V=length(mode);
%% initialize dual variable
Y=cell(1,V);
C=cell(1,V);
E=cell(1,V);
X=cell(1,V);
Maskt=cell(1,V);
B=cell(1,V);
G_t=cell(1,3);

G1=cell(1,V);
G2=cell(1,V);
G3=cell(1,V);
W1=cell(1,V);
W2=cell(1,V);
W3=cell(1,V);

s_X=[];

for k=1:1:V
    order=[mode(k):N 1:mode(k)-1];
    X_temp=permute(Xt,order);
    Mask_temp=permute(Mask,order);
    X_temp=reshape(X_temp,prod(n_t(order(1:d(k)))),[]);
    Mask_temp=reshape(Mask_temp,prod(n_t(order(1:d(k)))),[]);
    ind=tcu_rearrange(n_t,mode(k),d(k));
    if N-mode(k)+1>d(k)
        X{k}=X_temp(:,ind);
        Maskt{k}=Mask_temp(:,ind);
    else
        X{k}=X_temp(ind,:)';
        Maskt{k}=Mask_temp(ind,:)';
    end
end

Mask_c=mt_blk_comb(Maskt,L);

%% initialize
for k=1:1:V
    Y{k}=zeros(size(X{k}));
    E{k}=zeros(size(X{k}));
    C{k}=zeros(size(X{k},2));
    G1{k}=zeros(size(X{k},2));
    G2{k}=zeros(size(X{k},2));
    G3{k}=zeros(size(X{k},2));
    W1{k}=zeros(size(X{k},2));
    W2{k}=zeros(size(X{k},2));
    W3{k}=zeros(size(X{k},2));
    s_X=[s_X; size(X{k},1) size(X{k},2)/L];
    XX{k}=X{k}'*X{k};
end

for itr=1:1:100
    
    %% Update C

    for k=1:1:V
        C{k}=(3*beta/mu*eye(size(C{k}))+XX{k})\(XX{k}-X{k}'*E{k}+1/mu*(X{k}'*Y{k}+beta*(G1{k}+G2{k}+G3{k})-(W1{k}+W2{k}+W3{k})));
    end

    %% Update E

    for k=1:1:V
        B{k}=X{k}-X{k}*C{k}+1/mu*Y{k};
    end

    B_c=(mt_blk_comb(B,L));
    E_c=(prox_l21_Mask(B_c,lambda/mu,Mask_c));
    E=mt_blk_decomb(E_c,V,s_X);

    %% Update G
    C_t=(tr_blk_comb(C,L));
    W_t{1}=(tr_blk_comb(W1,L));
    W_t{2}=(tr_blk_comb(W2,L));
    W_t{3}=(tr_blk_comb(W3,L));
    if ndims(C_t)==3
        for k=1:1:3
            A=TRunfold_t(C_t+1/beta*W_t{k},k,2);
            [U,S,VV]=svd(A,'econ');
            Sv=diag(S);
            Sv=max(Sv-1/3/beta,0);
            DA=U*diag(Sv)*VV';
            G_t{k}=TRfold_t(DA,k,size(W_t{k}),3);
        end    
    else
        A=C_t+1/beta*W_t{k};
        [U,S,VV]=svd(A,'econ');
        Sv=diag(S);
        Sv=max(Sv-1/3/beta,0);
        G_t{1}=U*diag(Sv)*VV';
        G_t{2}=U*diag(Sv)*VV';
        G_t{3}=U*diag(Sv)*VV';
    end
    G1=tr_blk_decomb(G_t{1},V,s_X);
    G2=tr_blk_decomb(G_t{2},V,s_X);
    G3=tr_blk_decomb(G_t{3},V,s_X);
    
    %% Update Y V and W
    for k=1:1:V
        Y{k}=Y{k}+mu*(X{k}-X{k}*C{k}-E{k});
        W1{k}=W1{k}+beta*(C{k}-G1{k});
        W2{k}=W2{k}+beta*(C{k}-G2{k});
        W3{k}=W3{k}+beta*(C{k}-G3{k});
    end    
    
    %% update beta and rho
    mu=min(mu*1.1,10);
    beta=min(beta*1.1,10);
     
    err2=C_t-G_t{2};
    cc=norm(err2(:))^2/norm(C_t(:))^2;
    
    if cc<option.stopc
        break;
    end

end

end