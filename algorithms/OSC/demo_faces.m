%  Faces demo

paths = genpath('libs/ncut');

addpath(paths);

%% Load faces
faces_1 = load_faces('B01');
faces_2 = load_faces('B02');
faces_3 = load_faces('B03');

rng(1);

X = [faces_1(:,randperm(64)), faces_2(:,randperm(64)), faces_3(:,randperm(64))];
X = X/255;

corruption = 0;

w = randn(size(X)) * corruption;
X = X + w;

nbCluster = 3;

%% OSC

maxIterationbCluster = 200;
lambda_1 = .5;
lambda_2 = 5;
gamma_1 = 0.01;
gamma_2 = 0.01;
p = 1.1;
diag = 1;

tic;
[Z, funVal] = OSC(X, lambda_1, lambda_2, gamma_1, gamma_2, p, maxIterationbCluster, diag );
toc;

[oscclusters,~,~] = ncutW((abs(Z)+abs(Z')),nbCluster);

clusters = denseSeg(oscclusters,1);
plotClusters(clusters);

rmpath(paths);