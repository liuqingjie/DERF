function sviv = init_DERF(im,setting)

if nargin == 1
    setting.NT           = 3;  % NT==1：normalize_full,  NT==2： normalize_partial, NT==3：normalize_sift.
    setting.SQ           = 5; % the number of the rings
    setting.LI           = 3;
    setting.NI           = 3;
    setting.rs           = [0  2  4  6  8]; % radius of the rings
    setting.nthetas      = [1 8  8  8  8];%[1 8 8 8 8 8 8 8 8 8 8 8 8 8 8 ];
    setting.ndescriptors = 520; % dimensional of derf descriptor
    setting.HQ           = 8; % the number of gradient oirentations
    setting.assemblingmanner = 'multi'; %%% multi：multi-derf  ，single：single-derf NOT USED 

    setting.ASQ          = 3; % NOT USED
    setting.numofver     = 1; % when TF-DoG： the number of rotation directions for one grid point and one scale. NOTE USED
end

if size(im,3)~=1
    im = rgb2gray(im);
end

im = single(im);

fprintf(1,'-------------------------------------------------------\n');
fprintf(1,'0. init grid  +  gradient;\n');

grid=compute_grid(setting); 

L=compute_gradient(im,setting);% this is about 10-15% faster than a c implementation

sig_inc=sqrt(1.6^2-0.5^2);

L=smooth_layers(L,sig_inc);

fprintf(1,'1. compute DOG layers ');
tic;

H = mex_get_DOG(single(L),single(setting.SQ));

time_cl=toc;
fprintf(1,'is done in %f sec\n',time_cl);

sviv.H                = single(H);
sviv.gradientmanner   = 'T1';
sviv.assemblingmanner = 'multi';
sviv.gridmanner       = 'radial'; 
sviv.confun           = 'DOG';
sviv.numofver         = setting.numofver;
       
sviv.h                = size(im,1);
sviv.w                = size(im,2);
sviv.SQ               = setting.SQ;
sviv.HQ               = setting.HQ;

sviv.rs               = setting.rs;
sviv.nthetas          = setting.nthetas;
sviv.ndescriptors     = setting.ndescriptors;
sviv.HN               = size(grid, 1);
sviv.grid             = single(grid);
%sviv.ogrid            = ogrid; 
%sviv.csigma           = [1 1.2599 1.5874 2 2.5198 3.1748 4  5.03968 6.34960 8 10.07936 12.6992 16 20.1587368  25.3984168]; %这个参数在日后处理的时候，可以根据图像大小来设置这个参数！！！！！！！！！！！！！！
%sviv.ostable          = compute_orientation_shift(setting.HQ,1);
sviv.SQ               = setting.SQ;
sviv.LI               = setting.LI;
sviv.NT               = setting.NT;
end


function grid=compute_grid(setting)  

gs=sum(setting.nthetas);

grid(gs,3)=single(0);
count=0;

for i=1:setting.SQ 
    thetas=( [0:(setting.nthetas(i)-1)]/setting.nthetas(i) )*2*pi;
    grid(count+1:count+setting.nthetas(i),3)=setting.rs(i)*cos(thetas)';  
    grid(count+1:count+setting.nthetas(i),2)=setting.rs(i)*sin(thetas)';%  
    grid(count+1:count+setting.nthetas(i),1)=i;    
    count=count+setting.nthetas(i);
end
end 

function L = compute_gradient(im,setting)

hf = gaussian_1d(0.5, 0, 5);
vf = hf';

im=conv2(im, hf, 'same');
im=conv2(im, vf, 'same');

f1 = [-0.5,0,0.5];
f2 = f1';

dx=conv2(im, f1, 'same');
dy=conv2(im, f2, 'same');

[h,w]=size(im);
L = zeros(h,w, setting.HQ);

for l=0:setting.HQ-1
    th=2*l*pi/setting.HQ;
    kos=cos(th);
    zin=sin(th);
    L(:,:,l+1)=max(kos*dx+zin*dy, 0);%%  
end 
end
