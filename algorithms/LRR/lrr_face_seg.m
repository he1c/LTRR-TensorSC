function [] = lrr_face_seg()
data = loadmatfile('yaleb10.mat');
X = data.X;
gnd = data.cids;
K = max(gnd);
tic;
%run lrr
Z = solve_lrr(X,0.18);

%post processing
[U,S,V] = svd(Z,'econ');
S = diag(S);
r = sum(S>1e-4*S(1));
U = U(:,1:r);S = S(1:r);
U = U*diag(sqrt(S));
U = normr(U);
L = (U*U').^4;

% spectral clustering
D = diag(1./sqrt(sum(L,2)));
L = D*L*D;
[U,S,V] = svd(L);
V = U(:,1:K);
V = D*V;
idx = kmeans(V,K,'emptyaction','singleton','replicates',20,'display','off');
acc =  compacc(idx,gnd);
disp(['seg acc=' num2str(acc)]);
toc;