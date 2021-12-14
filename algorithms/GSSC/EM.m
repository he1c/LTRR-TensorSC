function [G,Uhat] = EM(XO,K,r,params,Uinit)
% ====================================================================
% Estimates the union of K r-dimensional subspaces that fits the
% incomplete data matrix XO according to the EM algorithm in
%
%   D. Pimentel, L. Balzano and R. Nowak,
%   On the Sample Complexity of Subspace Clustering with Missing Data,
%   IEEE SSP, 2014.
% 
% Input:
%   
%   XO = dxN data matrix.  Zeros correspond to missing entries.
%   K = number of subspaces.
%   r = dimension of subspaces.
%   params = vector with two parameters: tolerance and maximum number of
%            iterations.
%   Uinit = initial subspaces estimates.
%
% Output:
%
%   G = clustering of the datapoints.
%   Uhat = collection of K r-dimensional subspaces that fit XO.
%
%
% Written by: Daniel Pimentel
% email: pimentelalar@wisc.edu
% Created: 2015
% =====================================================================



% =========================== EM Algorithm ===========================
tol = params(1);
maxIter = params(2);

% == Initialize estimates ==
h = initializeh(XO,K,r,tol);
h.W = Uinit;

dif = tol + 1;
it = 0;
% =========================== Run EM ===========================
while it < maxIter && dif >= tol
    it = it+1;
    
    % Keep track of our previous estimates to check for convergence
    %prevh = h;
    prevX = h.X;
    
    % Run EM
    [PL,E] = Estep(XO,h);
    h = Mstep(PL,E,h);
    
    % ===== Find parameter difference as convergence criteria =====
    dif = norm(prevX-h.X)./norm(h.X);
    
    %if mod(it,10)==1, fprintf('\n \t \t %d    ',it); else fprintf('%d    ',it); end
end

%fprintf('\n');
G = h.G;
Uhat = h.W;
end

function [PL,E] = Estep(XO,h)
% ============================ E-step ============================

PL = PosteriorLikelihood(XO,h);
E = Expectations(XO,h);
end

function E = Expectations(XO,h)
% =============== Find E[x], E[yy'], E[xy'] and E[xx'] ===============

E = struct();
E.Y = cell(h.N,h.K);
E.X = cell(h.N,h.K);
E.YY = cell(h.N,h.K);
E.XY = cell(h.N,h.K);
E.XX = cell(h.N,h.K);

for i=1:h.N,
    idxO = h.Idx.O{i};
    idxM = h.Idx.M{i};
    xio = XO(idxO,i);
    for k=1:h.K,
        hWk = h.W{k};
        hWkOi = hWk(idxO,:);
        hWkMi = hWk(idxM,:);
        
        E.Y{i,k} = zeros(h.r,1);
        E.X{i,k} = zeros(h.d,1);
        
        E.Y{i,k} = (hWkOi'*hWkOi + h.s2*eye(h.r))\hWkOi'*xio;
        E.X{i,k}(idxO) = xio;
        E.X{i,k}(idxM) = hWkMi*E.Y{i,k};
        
        % Find cov(y), cov(x^m,y) to later find E[yy]] and E[xm y']
        covXmY = h.s2*hWkMi/(hWkOi'*hWkOi + h.s2*eye(h.r));
        
        E.YY{i,k} = h.s2*eye(h.r)/(hWkOi'*hWkOi + h.s2*eye(h.r)) + E.Y{i,k}*E.Y{i,k}';
        E.XY{i,k}(idxO,:) = xio*E.Y{i,k}';
        E.XY{i,k}(idxM,:) = covXmY + E.X{i,k}(idxM) * E.Y{i,k}';
        
        
    end
end

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

function p = gausspdf(x,mu,C)
% ================ Gaussian probability density function ================

p = exp(-1/2*(x-mu)'/C*(x-mu))/sqrt(abs(det(C)))/((2*pi)^(length(x)/2));
if isnan(p)
    p = 0;
end
end

function Idx = indices(Xw)
% == Determine indices of the observed and unobserved entries ==

N = size(Xw,2);

Idx = struct();
Idx.O = cell(N,1);
Idx.M = cell(N,1);
for i=1:N,
    Idx.O{i} = find(Xw(:,i)~=0);
    Idx.M{i} = find(Xw(:,i)==0);
end
end

function h = initializeh(XO,K,r,s2)
% ===================== Initialize parameters =====================

h = struct();
[h.d,h.N] = size(XO);
h.K = K;
h.r = r;
h.s2 = s2;

h.X = XO;
h.W = farthestW(XO,K,r);
h.Rho = 1/K * ones(1,K);

% == Determine indices once and for all for efficiency ==
h.Idx = indices(XO);
end

function h = Mstep(PL,E,h)
% ============================ M-step ============================

% Find our new h.W
h.W = cell(1,h.K);
for k=1:h.K,
    num = 0;
    den = 0;
    for i=1:h.N,
        num = num + PL(i,k)*E.XY{i,k};
        den = den + PL(i,k)*E.YY{i,k};
    end
    h.W{k} = num/den;
end


% Cluster
[~,h.G] = max(PL,[],2);

% Find our new h.X
h.X = zeros(h.d,h.N);
for i=1:h.N,
    h.X(:,i) = E.X{i,h.G(i)};
end
end

function PL = PosteriorLikelihood(XO,h)
% == Find new Posterior Likelihood according to the new parameters h ==

PL = zeros(h.N,h.K);
for i=1:h.N,
    idxO = h.Idx.O{i};
    xio = XO(idxO,i);
    for k=1:h.K,
        hWk = h.W{k};
        hWkOi = hWk(idxO,:);
        hCkOi = hWkOi*hWkOi' + h.s2*eye(length(idxO));
        PL(i,k) = gausspdf(xio,0,hCkOi);
    end
    if sum(PL(i,:))==Inf
        PL(i,:) = PL(i,:)==Inf;
    end
    if sum(PL(i,:))==0,
        PL(i,:) = 1;
    end
    PL(i,:) = PL(i,:).*h.Rho;
    PL(i,:) = PL(i,:)/sum(PL(i,:));
end

end




