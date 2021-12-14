addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('utils'));

%% step 1 read data and process data 
load('.\data\extendyaleb.mat');
X=X/255.0; 

%% step 2 construct the dictionary  
opts.denoising_flag=0; % set the flag whether we use R-TPCA to construct the dictionary 
% (1 denotes we use R-TPCA; 0 deonotes we do not use)

if opts.denoising_flag% if we use R-TPCA to construct the dictionary, we set its parameters
    [n1,n2,n3]=size(X);
    opts.lambda = 1/sqrt(max(n1,n2)*n3);
    opts.mu = 1e-4;
    opts.tol = 1e-8;
    opts.rho = 1.2;
    opts.max_iter = 800;
    opts.DEBUG = 0; %% whether we debug the algorithm
end    

% run the dictionary construction algorithm
[LL,V] = dictionary_learning(X,opts);
 

%% Step 3 test R-TLRR
max_iter=800;
DEBUG = 0; %% do not output the convergence behaviors at each iteration
[Z,tlrr_E,Z_rank,err_va ] = Tensor_LRR(X,LL,max_iter,DEBUG);
Z=tprod(V,Z); %% recover the real representation



%% step 4 cluster data
t=0.01;
[ mean_nmi] = ncut_clustering(Z, label',t );
fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi(1),mean_nmi(2),mean_nmi(3));



