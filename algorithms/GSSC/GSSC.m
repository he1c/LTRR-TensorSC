function [Ghat,Uhat] = GSSC(XO,K,r,params,Uinit)

% ====================================================================
% Estimates the union of K r-dimensional subspaces that fits the
% incomplete data matrix XO according to the Gropu-Sparse
% Subspace Clustering (GSSC) algorithm in
%
%   D. Pimentel, L. Balzano, R. Marcia, R. Nowak and R. Willett,
%   Group-Sparse Subspace Clustering with Missing Data,
%   IEEE SSP, 2016.
% 
% Input:
%   
%   XO = dxN data matrix.  Zeros correspond to missing entries.
%   K = number of subspaces.
%   r = dimension of subspaces.
%   params = vector with three parameters: tolerance, maximum number of
%            iterations and lambda (optimization penalty parameter).
%   Uinit = initial subspaces estimates.
%
% Output:
%
%   Ghat = clustering of the datapoints.
%   Uhat = collection of K r-dimensional subspaces that fit XO.
%
%
% Written by: D. Pimentel, R. Marcia and R. Willet
% email: pimentelalar@wisc.edu
% Created: 2016
% =====================================================================


%Parameters
tol = params(1);
maxIter = params(2);
lambda = params(3);
O = XO~=0;
[d,N] = size(XO);

% =============== Initialize ===============
% Setup U in the structure we want
U = zeros(d,K*r);
for k=1:K,
    U(:,(k-1)*r+1:k*r) = Uinit{k};
end
U = U/norm(U,'fro');
%Initialize V according to U
cvx_begin quiet
    cvx_precision low
    variable V(N,K*r);
    func = norm(O.*(U*V'-XO),'fro');
    pnlty = lambda * sum(norms(reshape(V',[r,N*K]),2,1));
    minimize(func + pnlty);
cvx_end

%=============== Start iterations ===============
obj = evalObj(O,U,V,XO,N,K,r,lambda);
objChange = realmax;
it = 0;
while objChange>tol && it<=maxIter,
    
    objPrev = obj;
    it = it+1;
    
    %%==============Fix V and solve for U=============
    cvx_begin quiet
    cvx_precision low
    variable U(d,K*r);
    func = norm(O.*(U*V'-XO),'fro');
    minimize(func);
    norm(U,'fro')<=1;
    cvx_end
    
    %==============Fix U and solve for V=============
    cvx_begin quiet
    cvx_precision low
    variable V(N,K*r);
    func = norm(O.*(U*V'-XO),'fro');
    pnlty = lambda * sum(norms(reshape(V',[r,N*K]),2,1));
    minimize(func + pnlty);
    cvx_end
    
    obj = evalObj(O,U,V,XO,N,K,r,lambda);
    objChange = objPrev - obj;
    
    %fprintf('iter = %d \t objChange = %1.2g \t obj = %1.2g \t objPrev = %1.2g \t lambda = %1.2g \n',...
    %    it,objChange,obj,objPrev,lambda);
    
end

% =============== Cluster datapoints ===============
Ghat = cluster(V,K);
Uhat = cell(K,1);
for k=1:K,
    Uhat{k} = U(:,(k-1)*r+1:k*r);
end

end







function obj = evalObj(O,U,V,XO,N,K,r,lambda)

obj = norm(O.*(U*V'-XO),'fro');
for i = 1:N
    for k = 1:K
        obj = obj+norm(V(i,(k-1)*r+1:k*r))*lambda;
    end
end
end

function G = cluster(V,K)
[N,Kr] = size(V);
r = Kr/K;
normsV = zeros(K,N);
for k=1:K,
    Vk = V(:,(k-1)*r+1:k*r);
    normsV(k,:) = diag(Vk*Vk');
end

[~,G] = max(normsV);
G = G';

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










