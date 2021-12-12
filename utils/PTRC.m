function Xt=PTRC(M,Mask,I,option)

stopc=option.stopc;
d=option.d;
alpha=option.beta;
r=option.r;
maxitr=option.maxitr;
debug=option.debug;

N=ndims(M);
J=size(M);
X=cell(N,1);

%% Get L TR unfolding matrices

%%===initialize===
for k=1:1:N
    order=[k:N 1:k-1];
    W{k}=1*rand(prod(J(order(1:d))),r(k));
    H{k}=1*rand(r(k),prod(J(order(d+1:end))));  
end

Xt=M;

Xt_pre=0;

count=0;

for kk=1:1:maxitr
       
    for k=1:N
       order=[k:N 1:k-1];
       X_temp=permute(Xt,order);
       X{k}=reshape(X_temp,prod(J(order(1:d))),[]);
       W{k}=X{k}*H{k}';
       %H{k}=inv(W{k}'*B{k}*W{k})*W{k}'*B{k}*X{k};
       H{k}=inv(W{k}'*W{k}+1e-5*eye(size(W{k},2)))*W{k}'*X{k};
       X{k}=W{k}*H{k};
    end
    
    Xt=0;
    
    for k=1:N
       order=[k:N 1:k-1];
       X_temp=reshape(X{k},J(order));
       Xt=Xt+alpha(k)*(1-Mask).*ipermute(X_temp,order);
    end
    
    Xt=Xt+Mask.*M;
    
    E=Xt-Xt_pre;
    
    err=norm(E(:))/norm(Xt(:));
    
    if err<stopc
        count=count+1;
        if count>2
            if debug
                err1=Xt-I;
                fprintf('Iter %.0f, Diff %.2f\n',kk,norm(err1(:)));
            end
            break;
        end
    else
        count=0;
    end
    
    Xt_pre=Xt;
    
    if debug&&mod(kk,10)==0
        err1=Xt-I;
        fprintf('Iter %.0f, Diff %.2f\n',kk,norm(err1(:)));
    end
     

end