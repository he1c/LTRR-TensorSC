%% Code for synthetic data experiment: Phase transitions for different rank r and observation rate p(%). (Fig.2)
clear
addpath(genpath(pwd))
 
n_cluster=10;  % cluster number
sigma=0.04;  % Gaussian noise var
r=10; % TR rank r
pp=0.5; % pp*% observation rate
n_t=[10,10,10,100];  % full tensor size (data per cluster: n_t(end)/n_cluster);
N=length(n_t);
d=ceil(N/2);
n_t_mod=n_t;
n_t_mod(end)=floor(n_t(end)/n_cluster);

fprintf('Data model: subtensor size:[%d %d %d], data per cluster:%d, TR rank:%d, observation rate:%.2f\n',n_t(1),n_t(2),n_t(3),n_t_mod(end),r,pp);
fprintf('Data generation...');

%% label initialization
Gstar=kron(1:n_cluster,ones(n_t(end)/n_cluster,1));
Gstar=Gstar(:);

%% MMTSC option
option.ndata=n_t(end);
option.mu=1e-2;
option.beta=10e-2;
option.stopc=1e-3;
option.rho=1.1;

%% Data generation
TRrank=r*ones(N,1);
Xtg=[];
for i=1:1:n_cluster
    X_temp=gen_synthetic_TR(n_t_mod,TRrank);
    Xtg=cat(N,Xtg,X_temp);
end
Xgt=Xtg/(max(Xtg(:)));
G=sigma*randn(size(Xtg));
Xt=Xtg+G;
X_t=reshape(Xt,n_t);

%% Create Mask and Data (pp*% observation rate)
Mask_t=zeros(n_t);
omega=find(rand(prod(n_t),1)<pp);
Mask_t(omega)=1;
MissM_t=X_t.*Mask_t;

fprintf('Done\n');

%% ================No completion================
%% MMTSC-4
option.d=[2,2,2,2];
option.view=[1,2,3,4];
option.lambda=0.4;
[C_t,~]=MMTSC(MissM_t,Mask_t,option,Gstar);
GG=sum(abs(C_t),3);
C_MMTSC_4 = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-4 acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_4));

%% MMTSC-2
option.d=[2,2];
option.view=[1,2];
option.lambda=0.2;
[C_t,~]=MMTSC(MissM_t,Mask_t,option,Gstar);
GG=sum(abs(C_t),3);
C_MMTSC_2 = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-2 acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_2));

%% MMTSC-1
option.d=2;
option.view=1;
option.lambda=0.1;
[C_t,~]=MMTSC(MissM_t,Mask_t,option,Gstar);
GG=sum(abs(C_t),3);
C_MMTSC_1 = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-1 acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_1));

%% SSC_PZF
X_temp=reshape(MissM_t,prod(n_t(1:N-1)),[]);
params = [1e-4,1e+3,.1];   
cr=30;
[~,C_temp] = SSC_EWZF(X_temp,n_cluster,cr,params);
C_SSC_PZF= SpectralClustering(abs(C_temp)+abs(C_temp')-2*abs(diag(diag(C_temp))),n_cluster);
fprintf('SSC-PZF acc:%.4f\n',evalAccuracy(Gstar,C_SSC_PZF));

%% ==============Tensor completion================
sk=[];
for k=1:N
    order=[k:N 1:k-1];
    M_temp=reshape(MissM_t,prod(n_t(order(1:d))),[]);
    sk=[sk max(ceil((min(size(M_temp)))*0.2*sqrt(pp)),max(floor(sqrt(n_t(end))*2),3))];
end
optiontc.d = d;
optiontc.beta = 1/N*ones(N,1);
optiontc.r = sk;
optiontc.lambda = 1;
optiontc.stopc = 1e-3;
optiontc.maxitr  = 300;
optiontc.debug= 0 ;
I_hat=PTRC(MissM_t,Mask_t,[],optiontc);

%% MMTSC-2+TC
option.d=[2,2];
option.view=[1,2];
option.lambda=0.2;
[C_t,~]=MMTSC(I_hat,Mask_t>-1,option,Gstar);
GG=sum(abs(C_t),3);
C_MMTSC_2_TC = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-2+TC acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_2_TC));

%% MMTSC-1+TC
option.d=2;
option.view=1;
option.lambda=0.1;
[C_t,~]=MMTSC(I_hat,Mask_t>-1,option,Gstar);
GG=sum(abs(C_t),3);
C_MMTSC_1_TC = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-1+TC acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_1_TC));

%% SSC+TC
X_temp=reshape(I_hat,prod(n_t(1:N-1)),[]);
[~,C_temp]=SSC(X_temp,0,false,30,false,1,Gstar);
C_temp(isnan(C_temp))=0;
C_SSC_TC = SpectralClustering(abs(C_temp)+abs(C_temp')-2*abs(diag(diag(C_temp))),n_cluster);
fprintf('SSC+TC acc:%.4f\n',evalAccuracy(Gstar,C_SSC_TC));

%% LRR+TC
[~,C_temp]=LRR(X_temp,Gstar,1.5);
C_temp(isnan(C_temp))=0;
C_LRR_TC = SpectralClustering(abs(C_temp)+abs(C_temp')-2*abs(diag(diag(C_temp))),n_cluster);
fprintf('LRR+TC acc:%.4f\n',evalAccuracy(Gstar,C_LRR_TC));













