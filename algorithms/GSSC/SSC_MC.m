function [Ghat,Uhat] = SSC_MC(XO,K,r,params)

% ====================================================================
% Estimates the union of K r-dimensional subspaces that fits the
% incomplete data matrix XO according to the Matrix Completion plus
% Sparse Subspace Clustering (MC+SSC) algorithm in
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
%            iterations
%
% Output:
%
%   Ghat = clustering of the datapoints.
%   Uhat = collection of K r-dimensional subspaces that fit XO.
%
%
% Written by: D. Pimentel, but mostly using code from
%             Ehsan Elhamifar and Chong You (for the SSC part) and
%             Yin Zhang and Zaiwen Wen (for the MC part, using LMAFIT)
% email: pimentelalar@wisc.edu
% Created: 2016
% =====================================================================

tol = params(1);
maxIter = params(2);
Xmc = callLMaFit(XO,[K*r,tol,maxIter]);

Ghat = ssc(Xmc,K,r,'ADMM',0);

Uhat = cell(K,1);
for k=1:K,
    XOk = XO(:,Ghat==k);
    Xk = callLMaFit(XOk,[r,tol,maxIter]);
    [Uk,~,~] = svd(Xk);
    Uhat{k} = Uk(:,1:r);
end

end