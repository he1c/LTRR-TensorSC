clear all;

addpath(genpath(pwd))

n_cluster=10;
sigma=0.001;
pp=0.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
video_name={'birds','boats','bottle','chopper','cyclists','flock','freeway','hockey','jump','landing','ocean','peds','rain','skiing','surf','surfers','traffic','zodiac'};
video_framenum=[71,31,31,100,30,246,44,50,81,50,176,170,229,111,87,41,190,189];
selected_classes = sort(randsample(18 , n_cluster)) ;
X = []; Gstar = [] ;
    
for m = 1:n_cluster         % forming the data
    % selected_classes(m)
    frame_num=min(video_framenum(selected_classes(m)),50);
    for kk=1:1:frame_num
        Image=imread(['UCSD\' video_name{selected_classes(m)} '\frame_' num2str(kk) '.jpg']);
        Image_s=imresize(Image,[36 56]);
        X=[X Image_s(:)];
    end
    Gstar = [Gstar m*ones(1,frame_num)] ;         % Xlabels contains the true labels of the data
end

%% reshape and add noise 
Xtg=double(reshape(X,36,56,[]))/255;
G=sqrt(sigma)*randn(size(Xtg));
Xt=Xtg+G;

%% shuffle the frames
idx=randperm(size(Xt,3));
Xt=Xt(:,:,idx);
Gstar=Gstar(idx);

n_t=[6,6,7,8,size(Xt,3)];
N=length(n_t);
d=ceil(N/2);
X_t=reshape(Xt,n_t);

%% option
option.ndata=n_t(end);
option.mu=1e-2;
option.beta=10e-2;
option.stopc=1e-3;
option.rho=1.1;
   
%% Create partially observed Data
Mask_t=zeros(n_t);
omega=find(rand(prod(n_t),1)<pp);
Mask_t(omega)=1;
MissM_t=X_t.*Mask_t;

%% ============No completion================
%% MMTSC-2
option.d=[3,3];
option.view=[1,2];
option.lambda=0.2;
[C_t,~]=MMTSC(MissM_t,Mask_t,option,Gstar);
GG=double(sum(abs(C_t),3));
C_MMTSC_2 = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-2 acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_2));

%% MMTSC-1
option.d=3;
option.view=1;
option.lambda=0.1;
[C_t,~]=MMTSC(MissM_t,Mask_t,option,Gstar);
GG=double(sum(abs(C_t),3));
C_MMTSC_1 = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-1 acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_1));

%% SSC-PZF
X_temp=reshape(MissM_t,prod(n_t(1:N-1)),[]);
params = [1e-4,1e+3,.1];   %SSC_EWZF
cr=50;
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
optiontc.d=d;
optiontc.beta=1/N*ones(N,1);
optiontc.r=sk;
optiontc.lambda=1;
optiontc.stopc=1e-3;
optiontc.maxitr  = 300;
optiontc.debug=0;
I_hat=PTRC(MissM_t,Mask_t,[],optiontc);

%% MMTSC-2+TC
option.d=[3,3];
option.view=[1,2];
option.lambda=0.2;
[C_t,~]=MMTSC(I_hat,Mask_t>-1,option,Gstar);
GG=double(sum(abs(C_t),3));
C_MMTSC_2_TC = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-2+TC acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_2_TC));

%% MMTSC-1+TC
option.d=3;
option.view=1;
option.lambda=0.1;
[C_t,~]=MMTSC(I_hat,Mask_t>-1,option,Gstar);
GG=double(sum(abs(C_t),3));
C_MMTSC_1_TC = SpectralClustering(GG+GG'-2*diag(diag(GG)),n_cluster);
fprintf('MMTSC-1+TC acc:%.4f\n',evalAccuracy(Gstar,C_MMTSC_1_TC));

%% SSC+TC
X_temp=reshape(I_hat,prod(n_t(1:N-1)),[]);
[~,C_temp]=SSC(X_temp,0,false,100,false,1,Gstar);
C_SSC_TC = SpectralClustering(abs(C_temp)+abs(C_temp'),n_cluster);
fprintf('SSC+TC acc:%.4f\n',evalAccuracy(Gstar,C_SSC_TC));

%% LRR+TC
[~,C_temp]=LRR(X_temp,Gstar,1.5);
C_LRR_TC= SpectralClustering(abs(C_temp)+abs(C_temp')-2*abs(diag(diag(C_temp))),n_cluster);

%% OSC+TC
maxIterations = 200;
lambda_1 = 0.01;
lambda_2 = 1;
gamma_1 = 0.01;
gamma_2 = 0.01;
p = 1.1;
X_OSC = normalize(X_temp);
[Z, funVal] = OSC(X_OSC, lambda_1, lambda_2, gamma_1, gamma_2, p, maxIterations);
C_OSC_TC = SpectralClustering(abs(Z)+abs(Z')-2*abs(diag(diag(Z))),n_cluster);
fprintf('OSC+TC acc:%.4f\n',evalAccuracy(Gstar,C_OSC_TC));

%% TLRR+TC
I_hat_3=reshape(I_hat,size(Xt));
X_TLRR=permute(I_hat_3,[1 3 2]);
opts.denoising_flag=0; % set the flag whether we use R-TPCA to construct the dictionary 
[LL,V] = dictionary_learning(X_TLRR,opts);
max_iter=800;
DEBUG = 0; 
[Z,tlrr_E,Z_rank,err_va ] = Tensor_LRR(X_TLRR,LL,max_iter,DEBUG);
Z=tprod(V,Z); 
[~,C_TSVDLRR_TC] = ncut_clustering(Z, Gstar',0.01);
fprintf('TSVD-TLRR+TC acc:%.4f\n',evalAccuracy(Gstar,C_TSVDLRR_TC));
      










