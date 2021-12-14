% Video demo

paths = genpath('libs/ncut');

addpath(paths);

%% Load video
cellspan = 'D5:G67';
vid_name = 'confederate_honey';

frame_data = xlsread('../data/video/frame_data.xlsx',1,cellspan);
frame_data = frame_data(~any(isnan(frame_data),2),:);

vid_obj = VideoReader(['../data/video/' vid_name '.mp4']);

corruption = 0;

nbCluster = 3;

height = 96;
width = 129;
vidsize = [height, width];

rng(1);

i = 14;

startFrame = max(frame_data(i,1), frame_data(i,2) - 50);
endFrame = min(frame_data(i,4), frame_data(i,3) + 50);
trans_1_frame = frame_data(i,2);
trans_2_frame = frame_data(i,3);

truth = [ones(1,trans_1_frame - startFrame), 2*ones(1, trans_2_frame - trans_1_frame), 3*ones(1, endFrame - trans_2_frame)]';

length = endFrame - startFrame;

X = zeros(height*width,length);

for k=1:length
    X(:,k) = reshape(imresize(rgb2gray(read(vid_obj,startFrame + k+1)),vidsize),height*width,1);
end

X = double(X)/255;

w = randn(height*width,length) * corruption;
X = X + w;

X = normalize(X);

 %% OSC

maxIterations = 200;
lambda_1 = 0.01;
lambda_2 = 1;
gamma_1 = 0.01;
gamma_2 = 0.01;
p = 1.1;

tic;
[Z, funVal] = OSC(X, lambda_1, lambda_2, gamma_1, gamma_2, p, maxIterations);
toc;

[oscclusters,~,~] = ncutW((abs(Z)+abs(Z')),nbCluster);

clusters = denseSeg(oscclusters,1);
plotClusters(clusters);

rmpath(paths);