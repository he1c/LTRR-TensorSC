function [err,time] = runExperiment(d,K,r,Nk,ell,params)

% ====================================================================
% Sample code to replicate the experiments in
%
%   D. Pimentel, L. Balzano, R. Marcia, R. Nowak and R. Willett,
%   Group-Sparse Subspace Clustering with Missing Data,
%   IEEE SSP, 2016.
%
% Written by: D. Pimentel.
% email: pimentelalar@wisc.edu
% Created: 2016
% =====================================================================


% Generate data
[X,Ustar,Gstar] = gendata(d,r,K,Nk);
err = zeros(5,1);
time = zeros(5,1);

% Generate samples
O = zeros(d,K*Nk);
for i=1:K*Nk,
    O(randsample(d,ell),i) = 1;
end
XO = X.*O;

% ================== SSC-EWZF ==================
fprintf('Running SSC_EWZF...');

tic
[Ghat,U_EWZF] = SSC_EWZF(XO,K,r,params{1});
time(1) = toc;
c = correspondence(U_EWZF,Ustar);
err(1) = mean(Ghat~=c(Gstar)');
fprintf('\t Time = %1.1d \t Error = %1.1d. \n \t',time(1),err(1));

% ================== Simplex ==================
fprintf('Running MSC...');
tic
[Ghat,U_simplex] = MSC(XO,K,r,params{2},U_EWZF);
time(2) = toc;
c = correspondence(U_simplex,Ustar);
err(2) = mean(Ghat~=c(Gstar)');
fprintf('\t\t Time = %1.1d \t Error = %1.1d. \n \t',time(2),err(2));

% ================== GSSC ==================
fprintf('Running GSSC...');

tic
[Ghat,U_GSSC] = GSSC(XO,K,r,params{3},U_EWZF);
time(3) = toc;
c = correspondence(U_GSSC,Ustar);
err(3) = mean(Ghat~=c(Gstar)');
fprintf('\t\t Time = %1.1d \t Error = %1.1d. \n \t',time(3),err(3));

% ================== EM ==================
fprintf('Running EM...');
tic
[Ghat,U_EM] = EM(XO,K,r,params{4},U_EWZF);
time(4) = toc;
c = correspondence(U_EM,Ustar);
err(4) = mean(Ghat~=c(Gstar)');
fprintf('\t\t Time = %1.1d \t Error = %1.1d. \n \t',time(4),err(4));

% ================== MC ==================
fprintf('Running SSC_MC...');
tic
[Ghat,U_MC] = SSC_MC(XO,K,r,params{5});
time(5) = toc;
c = correspondence(U_MC,Ustar);
err(5) = mean(Ghat~=c(Gstar)');
fprintf('\t Time = %1.1d \t Error = %1.1d. \n',time(5),err(5));


end

function c = correspondence(Uhat,Ustar)

K = length(Uhat);

for k1=1:K,
    for k2=1:K,
        Residuals(k1,k2) = norm(Uhat{k1}-Projection(Uhat{k1},Ustar{k2}))/norm(Ustar{k2});
    end
end

[~,c] = min(Residuals);
end

function [X,Ustar,Gstar] = gendata(d,r,K,Nk)
% ================ generate synthetic data ================
sigma = 0;
Ustar = cell(1,2);
Gstar = zeros(K*Nk,1);
for k=1:K,
    Ustar{k} = orth(randn(d,r));
    
    Vk = randn(r,Nk);
    Xk = Ustar{k}*Vk + sigma*randn(d,Nk);
    X(:,(k-1)*Nk+1:k*Nk) = Xk;
    
    Gstar((k-1)*Nk+1:k*Nk) = k;
end
end

function hX = Projection(X,W)
% ========= Projection of X onto the subspace spanned by W =========

hX = W/(W'*W)*W'*X;

end



function U=farthestW(X,nc,cc)
%nn farthest insertion subspace initialization

no=sum(X.^2);
[dim N]=size(X);
rid=ceil(N*rand);
buf=5;
U = cell(1,nc);
for k=1:nc
    x=X(:,rid);
    sX=X-diag(x)*ones(dim,N);
    dd=sum(sX.^2);
    [td tdi]=sort(dd);
    [uu ss vv]=svd(X(:,tdi(1:cc+buf)),'econ');
    U{k}=uu(:,1:cc);
    md(:,k)=no-sum((U{k}'*X).^2);
    tmd=min(md,[],2);
    tmd=tmd/sum(tmd);
    
    rid=samplefromd(tmd);
end


end


function id=samplefromd(d)
%samples from density d
u=cumsum(d);
tu=u-rand;
tu(find(tu<0))=inf;
[ju id]=min(tu);

end












