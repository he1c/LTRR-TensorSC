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

function [groups,kerNS] = SpectralClustering(CKSym,n)

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

% K=n;
% n = size(kerNS,2);
% M = zeros(n,n,20);
% rand('state',123456789);
% for i=1:size(M,3)
%     inds = false(n,1);
%     while sum(inds)<K
%         j = ceil(rand()*n);
%         inds(j) = true;
%     end
%     M(:,:,i) = kerNS(inds,:);
% end
% groups2 = kmeans(kerNS,K,'emptyaction','singleton','start',M,'display','off');
