% ====================================================================
% Sample code to replicate the experiments in
%
%   D. Pimentel, L. Balzano, R. Marcia, R. Nowak and R. Willett,
%   Group-Sparse Subspace Clustering with Missing Data,
%   IEEE SSP, 2016.
%
% Written by: D. Pimentel.
% email: pimentelalar@wisc.edu
% Created: 2016
% =====================================================================

clear all; clc;
rand('state',sum(100*clock));

%==================== General Setup ====================
d = 100;        %ambient dimension
r = 25;         %subspaces dimensions
K = 5;          %number of subspaces

%=========== Particulars of the experiment ===========
Nk = 65;        %Columns per subspace
ell = 65;       %Observations per column

filename = ['results_d',num2str(d),'_r',num2str(r),'_K',num2str(K),'.mat'];

%=== Parameters for this General Setup, obtained by cross-validation ===
params{1} = [1e-4,1e+3,.001];   %SSC_EWZF
params{2} = .001;               %MSC
params{3} = [1e-3,100,.001];    %GSSC
params{4} = [1e-3,2*d];         %EM
params{5} = [1e-4,1e+3];        %SSC_MC

%=============== Run simulation ===============
fprintf('d = %d, r = %d, K = %d, Nk = %d, ell = %d. \t \n \t',d,r,K,Nk,ell);
[Err,Time] = runExperiment(d,K,r,Nk,ell,params);
save(filename);

