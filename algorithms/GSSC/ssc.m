function C = ssc(X,K,r,type,lambda)
if strcmp(type,'ADMM'),
    CoeffMat = admmLasso_mat_func(X,0,800,2*10^-4,200);
else
    CoeffMat = SparseCoefRecovery(X,0,type,lambda); %Options: Lasso, L1Noisy, etc.
end
AdjMat = BuildAdjacency(CoeffMat,r);
C = SpectralClustering(AdjMat,K);
end

%============== STUFF FOR CVX ==============
function CMat = SparseCoefRecovery(Xp,cst,Opt,lambda)
%--------------------------------------------------------------------------
% This function takes the D x N matrix of N data points and write every
% point as a sparse linear combination of other points.
% Xp: D x N matrix of N data points
% cst: 1 if using the affine constraint sum(c)=1, else 0
% Opt: type of optimization, {'L1Perfect','L1Noisy','Lasso','L1ED'}
% lambda: regularizartion parameter of LASSO, typically between 0.001 and 
% 0.1 or the noise level for 'L1Noise'
% CMat: N x N matrix of coefficients, column i correspond to the sparse
% coefficients of data point in column i of Xp
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------

if (nargin < 2)
    cst = 0;
end
if (nargin < 3)
    Opt = 'Lasso';
end
if (nargin < 4)
    lambda = 0.001;
end

D = size(Xp,1);
N = size(Xp,2);

for i = 1:N
    
    y = Xp(:,i);
    if i == 1
        Y = Xp(:,i+1:end);
    elseif ( (i > 1) && (i < N) )
        Y = [Xp(:,1:i-1) Xp(:,i+1:N)];        
    else
        Y = Xp(:,1:N-1);
    end
    
    % L1 optimization using CVX
    if cst == 1
        if ( strcmp(Opt , 'Lasso') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1,1);
            minimize( norm(c,1) + lambda * norm(Y * c  - y) );
            subject to
            sum(c) == 1;
            cvx_end;
        elseif ( strcmp(Opt , 'L1Perfect') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            Y * c  == y;
            sum(c) == 1;
            cvx_end;
        elseif ( strcmp(Opt , 'L1Noisy') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            norm( Y * c  - y ) <= lambda;
            sum(c) == 1;
            cvx_end;
        elseif ( strcmp(Opt , 'L1ED') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1+D,1);
            minimize( norm(c,1) );
            subject to
            [Y eye(D)] * c  == y;
            sum(c(1:N-1)) == 1;
            cvx_end;
        end
    else
        if ( strcmp(Opt , 'Lasso') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1,1);
            minimize( norm(c,1) + lambda * norm(Y * c  - y) );
            cvx_end;
        elseif ( strcmp(Opt , 'L1Perfect') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            Y * c  == y;
            cvx_end;
        elseif ( strcmp(Opt , 'L1Noisy') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            norm( Y * c  - y ) <= lambda;
            cvx_end;
        elseif ( strcmp(Opt , 'L1ED') )
            cvx_begin quiet;
            cvx_precision low
            variable c(N-1+D,1);
            minimize( norm(c,1) );
            subject to
            [Y eye(D)] * c  == y;
            cvx_end;
        end
    end
    
    % place 0's in the diagonals of the coefficient matrix
    if i == 1   
        CMat(1,1) = 0;
        CMat(2:N,1) = c(1:N-1);       
    elseif ( (i > 1) && (i < N) )
        CMat(1:i-1,i) = c(1:i-1);
        CMat(i,i) = 0;
        CMat(i+1:N,i) = c(i:N-1);
    else
        CMat(1:N-1,N) = c(1:N-1);
        CMat(N,N) = 0;
    end

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

%============== STUFF FOR ADMM ==============
function C2 = admmLasso_mat_func(Y,affine,alpha,thr,maxIter)
%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points
% Y: DxN data matrix
% affine: if true then enforce the affine constraint
% thr1: stopping threshold for the coefficient error ||Z-C||
% thr2: stopping threshold for the linear system error ||Y-YZ||
% maxIter: maximum number of iterations of ADMM
% C2: NxN sparse coefficient matrix
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

if (nargin < 2)
    % default subspaces are linear
    affine = false; 
end
if (nargin < 3)
    % default regularizarion parameters
    alpha = 800;
end
if (nargin < 4)
    % default coefficient error threshold to stop ADMM
    % default linear system error threshold to stop ADMM
    thr = 2*10^-4; 
end
if (nargin < 5)
    % default maximum number of iterations of ADMM
    maxIter = 200; 
end

if (length(alpha) == 1)
    alpha1 = alpha(1);
    alpha2 = alpha(1);
elseif (length(alpha) == 2)
    alpha1 = alpha(1);
    alpha2 = alpha(2);
end

if (length(thr) == 1)
    thr1 = thr(1);
    thr2 = thr(1);
elseif (length(thr) == 2)
    thr1 = thr(1);
    thr2 = thr(2);
end

N = size(Y,2);

% setting penalty parameters for the ADMM
mu1 = alpha1 * 1/computeLambda_mat(Y);
mu2 = alpha2 * 1;

if (~affine)
    % initialization
    A = inv(mu1*(Y'*Y)+mu2*eye(N));
    C1 = zeros(N,N);
    Lambda2 = zeros(N,N);
    err1 = 10*thr1; err2 = 10*thr2;
    i = 1;
    % ADMM iterations
    while ( err1(i) > thr1 && i < maxIter )
        % updating Z
        Z = A * (mu1*(Y'*Y)+mu2*(C1-Lambda2/mu2));
        Z = Z - diag(diag(Z));
        % updating C
        C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z+Lambda2/mu2);
        C2 = C2 - diag(diag(C2));
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 * (Z - C2);
        % computing errors
        err1(i+1) = errorCoef(Z,C2);
        err2(i+1) = errorLinSys(Y,Z);
        %
        C1 = C2;
        i = i + 1;
    end
    %fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end),err2(end),i);
else
    % initialization
    A = inv(mu1*(Y'*Y)+mu2*eye(N)+mu2*ones(N,N));
    C1 = zeros(N,N);
    Lambda2 = zeros(N,N);
    lambda3 = zeros(1,N);
    err1 = 10*thr1; err2 = 10*thr2; err3 = 10*thr1;
    i = 1;
    % ADMM iterations
    while ( (err1(i) > thr1 || err3(i) > thr1) && i < maxIter )
        % updating Z
        Z = A * (mu1*(Y'*Y)+mu2*(C1-Lambda2/mu2)+mu2*ones(N,1)*(ones(1,N)-lambda3/mu2));
        Z = Z - diag(diag(Z));
        % updating C
        C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z+Lambda2/mu2);
        C2 = C2 - diag(diag(C2));
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 * (Z - C2);
        lambda3 = lambda3 + mu2 * (ones(1,N)*Z - ones(1,N));
        % computing errors
        err1(i+1) = errorCoef(Z,C2);
        err2(i+1) = errorLinSys(Y,Z);
        err3(i+1) = errorCoef(ones(1,N)*Z,ones(1,N));
        %
        C1 = C2;
        i = i + 1;
    end
    %fprintf('err1: %2.4f, err2: %2.4f, err3: %2.4f, iter: %3.0f \n',err1(end),err2(end),err3(end),i);
end
end

function err = errorCoef(Z,C)
%--------------------------------------------------------------------------
% This function computes the maximum error between elements of two 
% coefficient matrices
% C: NxN coefficient matrix
% Z: NxN coefficient matrix
% err: infinite norm error between vectorized C and Z
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

err = max(max( abs(Z-C) ));
%err = norm(Z-C,'fro');
end

function err = errorLinSys(P,Z)
%--------------------------------------------------------------------------
% This function computes the maximum L2-norm error among the columns of the 
% residual of a linear system 
% Y: DxN data matrix of N data point in a D-dimensional space
% Z: NxN sparse coefficient matrix
% err: maximum L2-norm of the columns of Y-YZ 
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

[R,N] = size(Z);
if (R > N) 
    E = P(:,N+1:end) * Z(N+1:end,:);
    Y = P(:,1:N);
    Y0 = Y - E;
    C = Z(1:N,:);
else
    Y = P;
    Y0 = P;
    C = Z;
end

[Yn,n] = matrixNormalize(Y0);
M = repmat(n,size(Y,1),1);
S = Yn - Y * C ./ M;
err = sqrt( max( sum( S.^2,1 ) ) );
end

function lambda = computeLambda_mat(Y,P)
%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns the regularization constant of the L1 norm
% Y: DxN data matrix
% lambda: regularization parameter for lambda*||C||_1 + 0.5 ||Y-YC||_F^2
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

if (nargin < 2)
    P = Y;
end

N = size(Y,2);
T = P' * Y;
T(1:N,:) = T(1:N,:) - diag(diag(T(1:N,:)));
T = abs(T);
lambda = min(max(T,[],1));
end

function [Yn,n] = matrixNormalize(Y)

%--------------------------------------------------------------------------
% This function normalizes the columns of a given matrix 
% Y: DxN data matrix
% Yn: DxN data matrix whose columns have unit Euclidean norm
% n: N-dimensional vector of the norms of the columns of Y
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

for i = 1:size(Y,2)
    n(i) = norm(Y(:,i));
    Yn(:,i) = Y(:,i) ./ n(i);
end
end