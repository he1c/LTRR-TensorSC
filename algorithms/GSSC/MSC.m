function [Ghat,Uhat] = MSC(X,K,r,lambda,Uinit)

% ====================================================================
% Estimates the union of K r-dimensional subspaces that fits the
% incomplete data matrix XO according to the Mixture Subspace 
% Clustering (GSSC) algorithm in
%
%   D. Pimentel, L. Balzano, R. Marcia, R. Nowak and R. Willett,
%   Group-Sparse Subspace Clustering with Missing Data,
%   IEEE SSP, 2016.
% 
% Input:
%   
%   X = dxN data matrix.  Zeros correspond to missing entries.
%   K = number of subspaces.
%   r = dimension of subspaces.
%   lambda = optimization penalty parameter.
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


[d,n] = size(X);
lambda_z = lambda;
lambda_p = lambda;

Ginit = clusterUknown(X,Uinit);
Omega = X==0;
O = X~=0;

Pmat = zeros(n,K);
Zinit = zeros(d,n,K);
Vinit = zeros(r,n,K);
for k = 1:K
    cvx_begin quiet
    cvx_precision low
    variable Vk(r,n)
    func = norm(O.*(X-Uinit{k}*Vk) ,'fro');
    minimize(func)
    cvx_end
    Vinit(:,:,k) = Vk*diag(Ginit==k);
    Zinit(:,:,k) = Uinit{k}*Vk;
    Pmat(:,k) = (Ginit==k)*1;
end

Z = Zinit;
% alternating minimization

t = 1;

converged = false;
objValPrev = evalObj(X,Omega,Z,Pmat,lambda_z,lambda_p);

while ~converged && t <=200
   
   Pmat_prev = Pmat;
   Z_prev = Z;
   
   
   % update P's
   cvx_begin quiet
   cvx_precision low
   variable Pmat(n,K);
   Xhat = zeros(d,n);
   for k = 1:K
      Xhat = Xhat + Z_prev(:,:,k)*diag(Pmat(:,k));
   end
   func = norm(O.*(X-Xhat),'fro');
   minimize(func)
   for i = 1:n
      (Pmat(i,:))' == simplex(K);
   end
   cvx_end

   objValTemp = evalObj(X,Omega,Z_prev,Pmat,lambda_z,lambda_p);
   
   
   % update Z's
   [Z,numIter] = MultiSubspaceMC(X,Omega,Pmat,lambda_z,r,Z_prev,lambda_p);
   objVal = evalObj(X,Omega,Z,Pmat,lambda_z,lambda_p);
   
   objChange = objValPrev-objVal;
   
   pChange = norm(Pmat(:)-Pmat_prev(:),2);
   zChange = norm(Z(:)-Z_prev(:),2);
   
   objValPrev = objVal;
   t = t+1;
   if (pChange < 1e-6) && (zChange < 1e-4) || (abs(objChange) < 1e-2)
      converged = 1;
   end
   
end

Uhat = cell(K,1);
for k=1:K,
    [Uk,~,~] = svd(Z(:,:,k));
    Uhat{k} = Uk(:,1:r);
end
[~,Ghat] = max(Pmat,[],2);

end


%
function f = evalObj(X,Omega,Z,Pmat,lambda_z,lambda_p)

Xhat = zeros(size(Z(:,:,1)));
K = size(Z,3);
nucNorm = 0;
for k = 1:K
   
   Xhat = Xhat + Z(:,:,k)*diag(Pmat(:,k));
   S = svd(Z(:,:,k));
   nucNorm = nucNorm + sum(abs(S));
   
end
Xhat(Omega) = 0;
f = sum(sum((X-Xhat).^2)) + lambda_z*nucNorm + ...
   lambda_p*sum(abs(Pmat(:)));

end

%
function [Z,iter] = MultiSubspaceMC(X,Omega,Pmat,lambda_z,r,Z,lambda_p)

K = size(Pmat,2);
if nargin < 5
   Z = X;
end

objVal = evalObj(X,Omega,Z,Pmat,lambda_z,lambda_p);
converged = false;
iter = 0;
maxIter = 100;

while (~converged)&&(iter < maxIter)
   iter = iter + 1;
   Z_prev = Z;
   objValPrev = objVal;
   
   
   Yk = X;
   for j = 1:K
      Yk = Yk - Z_prev(:,:,j)*diag(Pmat(:,j));
   end
   
   grad_Z = zeros(size(Z));
   gradNorm2 = 0;
   for k = 1:K
      
      grad_Zk = -Yk*diag(Pmat(:,k));
      grad_Zk(Omega) = 0;
      
      grad_Z(:,:,k) = grad_Zk;
      gradNorm2 = gradNorm2 + norm(grad_Zk,'fro')^2;
      
   end
   
   alpha = 4;
   foundAlpha = false;
   while (~foundAlpha)&&(alpha > 1e-1);
      
      for k = 1:K
         
         Z(:,:,k) = SoftThSing(Z_prev(:,:,k)-alpha*grad_Z(:,:,k),alpha*lambda_z,r);
         
      end
      objValAlpha = evalObj(X,Omega,Z,Pmat,lambda_z,lambda_p);
      if objValAlpha <= objValPrev + sum(grad_Z(:).*(Z(:)-Z_prev(:))) + .5/alpha*sum((Z(:)-Z_prev(:)).^2); % - .25*alpha*gradNorm2
         foundAlpha = true;
      else
         alpha = alpha/2;
      end
      
   end
   
   objVal = evalObj(X,Omega,Z,Pmat,lambda_z,lambda_p);
   objChange = objValPrev - objVal;
   
   ZChange = norm(Z_prev(:)-Z(:),'fro');
   
   if (abs(objChange) < 1e-3) || (ZChange < 1e-3)
      converged = true;
   end
   

end


end



function Y = SoftThSing(X,lambda,r)

if nargin < 3
   [U,S,V] = svd(X);
   
   S = (S >= lambda).*(S - lambda);
   Y = U*S*V';
   
else
   [U,S,V] = svds(X,r);
   while min(S(:)) > lambda
      r = r*2;
      [U,S,V] = svds(X,r);
   end
   S = (S >= lambda).*(S - lambda);
   Y = U*S*V';
end

end


