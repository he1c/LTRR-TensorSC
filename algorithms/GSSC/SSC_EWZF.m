function [Ghat,Uhat,C] = SSC_EWZF(XO,K,r,params)

% ====================================================================
% Estimates the union of K r-dimensional subspaces that fits the
% incomplete data matrix XO according to the Entrywise Zero-fill
% Sparse Subspace Clustering (SSC-EWZF) algorithm in
%
%   C. Yang, D. Robinson and R. Vidal,
%   Sparse subspace clustering with missing entries, 
%   ICML, 2015.
% 
% Input:
%   
%   XO = dxN data matrix.  Zeros correspond to missing entries.
%   K = number of subspaces.
%   r = dimension of subspaces.
%   params = vector with three parameters: tolerance, maximum number of
%            iterations and lambda (optimization penalty parameter).
%
% Output:
%
%   Ghat = clustering of the datapoints.
%   Uhat = collection of K r-dimensional subspaces that fit XO.
%
%
% Written by: D. Pimentel, but mostly using code from
%             Ehsan Elhamifar and Chong You below
% email: pimentelalar@wisc.edu
% Created: 2016
% =====================================================================

tol = params(1);
maxIter = params(2);
lambda = params(3);

N = size(XO,2);
O = XO~=0;
C = zeros(N);

for i=1:N,
    %fprintf('I am in %d th column \n',i);
    oi = find(O(:,i));
    xoi = XO(oi,i);
    Xoi = XO(oi,:);
    cvx_begin quiet
        cvx_precision low
        variable c(N);
        func = norm(xoi-Xoi*c,'fro') + lambda*norm(c,1); %eqn(11)
        minimize(func)
        subject to
            c(i)==0
    cvx_end
    C(:,i) = c;
end

% %This is twice as fast, but runs out of memory for large d.
% cvx_begin quiet
% variable C(N,N);
% func = sum(abs(C(:)))*lambda + norm(O.*(XO-XO*C),'fro');
% minimize(func)
% subject to 
%   diag(C)==0
% cvx_end


AdjMat = BuildAdjacency(C,r);
[Ghat,~] = SpectralClustering(AdjMat,K);

Uhat = cell(K,1);
for k=1:K,
    XOk = XO(:,Ghat==k);
    Xk = callLMaFit(XOk,[r,tol,maxIter]);
    [Uk,~,~] = svd(Xk);
    Uhat{k} = Uk(:,1:r);
end

end

function CKSym = BuildAdjacency(CMat,K)
%--------------------------------------------------------------------------
% This function takes a NxN coefficient matrix and returns a NxN adjacency
% matrix by choosing only the K strongest connections in the similarity
% graph
% CMat: NxN coefficient matrix
% K: number of strongest edges to keep; if K=0 use all the coefficients
% CKSym: NxN symmetric adjacency matrix
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------

N = size(CMat,1);
CAbs = abs(CMat);
for i = 1:N
    c = CAbs(:,i);
    [PSrt,PInd] = sort(c,'descend');
    CAbs(:,i) = CAbs(:,i) ./ abs( c(PInd(1)) );
end

CSym = CAbs + CAbs';

if (K ~= 0)
    [Srt,Ind] = sort( CSym,1,'descend' );
    CK = zeros(N,N);
    for i = 1:N
        for j = 1:K
            CK( Ind(j,i),i ) = CSym( Ind(j,i),i ) ./ CSym( Ind(1,i),i );
        end
    end
    CKSym = CK + CK';
else
    CKSym = CSym;
end
end

function [groups, kerNS] = SpectralClustering(CKSym,n)

%--------------------------------------------------------------------------
% This function takes an adjacency matrix of a graph and computes the 
% clustering of the nodes using the spectral clustering algorithm of 
% Ng, Jordan and Weiss.
% CMat: NxN adjacency matrix
% n: number of groups for clustering
% groups: N-dimensional vector containing the memberships of the N points 
% to the n groups obtained by spectral clustering
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
% Modified @ Chong You, 2015
%--------------------------------------------------------------------------

warning off;
N = size(CKSym,1);
MAXiter = 1000; % Maximum number of iterations for KMeans 
REPlic = 20; % Number of replications for KMeans

% Normalized spectral clustering according to Ng & Jordan & Weiss
% using Normalized Symmetric Laplacian L = I - D^{-1/2} W D^{-1/2}

DN = diag( 1./sqrt(sum(CKSym)+eps) );
LapN = speye(N) - DN * CKSym * DN;
[~,~,vN] = svd(LapN);
kerN = vN(:,N-n+1:N);
normN = sum(kerN .^2, 2) .^.5;
kerNS = bsxfun(@rdivide, kerN, normN + eps);
groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
end