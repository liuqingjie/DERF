
setting.NT = 3;  %%% NT==1£ºnormalize_full,  NT==2£º normalize_partial, NT==3£ºnormalize_sift.
setting.SQ = 5; % the number of the rings
setting.LI = 3;
setting.NI = 3;
setting.rs=[0   2   4   6  8];
setting.nthetas=[1  8  8  8  8];%%%[1 8 8 8 8 8 8 8 8 8 8 8 8 8 8 ];
setting.ndescriptors=520; % 
setting.HQ=8; %%% the number of gradient oirentations
setting.gradientmanner='T0'; %%%T0  T1  T2   T3  different gradients
setting.assemblingmanner='multi'; %%% multi£ºmulti-derf  £¬single£ºsingle-derf¡£

setting.ASQ=3; %%% 
setting.gridmanner='radial'; %%% 
setting.confun='DOG'; %%% convolution kernel    DOG:DoG wavelet   Haar£ºHaar wavelet
setting.numofver=1;  %% when TF-DoG£º the number of rotation directions for one grid point and one scale.

im = imread('lena512.pgm');
if size(im,3)>1
    im = rgb2gray(im);
end

% initialize derf 
sviv=init_DERF(im, setting);
params = [sviv.ndescriptors,sviv.HN,sviv.h,sviv.w,sviv.SQ,sviv.HQ,sviv.HQ,65];
fprintf(1,'-------------------------------------------------------\n');

%% extract derf descriptor from one specific point 
y = 150;
x = 155;
coord = int32([y, x]);
fprintf(1,'2. extracting derf descriptor from [%d,%d] ',y,x);
tic;
descriptor = mex_assembling_descriptor(sviv.H,single(params),sviv.grid,coord);
t1 = toc;
fprintf(1,'is done in %f sec\n',t1);

%% extract derf descriptors from a list of points
[h,w] = size(im);
% generate points
patchSize = 30;
overlap  = 15; 

co = getpatches(h,w,patchSize,overlap);
nSample = size(co,2);
coord = zeros(nSample,2);
coord(:,1) = floor((co(4,:)' + co(2,:)')/2);
coord(:,2) = floor((co(3,:)' + co(1,:)')/2);

fprintf(1,'3. extracting %d derf descriptors ',nSample);
tic;
descriptors = mex_assembling_descriptors(sviv.H,single(params),sviv.grid,single(coord));%each column is a descriptor
t2=toc;
fprintf(1,'is done in %f sec\n',t2);

%% extract all derf descriptors from an image
fprintf(1,'4. extracting all (%d*%d) derf descriptors ',h,w);
tic;
all_descriptors = mex_assembling_all_descriptors(sviv.H,single(params),sviv.grid);% it will take about 1.4 sec to process a 256*256 image
t3=toc;
fprintf(1,'is done in %f sec\n',t3);
fprintf(1,'-------------------------------------------------------\n');
