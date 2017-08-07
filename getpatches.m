function coordinates_of_patches = getpatches(h,w,patch_size,over_lap)

step = patch_size - over_lap;

[X2,Y2] = meshgrid(patch_size:step:w,patch_size:step:h);

X2 = X2(:);
Y2 = Y2(:);
X1 = X2 - patch_size+1;
Y1 = Y2 - patch_size+1;

len = length(X2);

coordinates_of_patches = zeros(4,len);

coordinates_of_patches(1,:) = X1(1:len);
coordinates_of_patches(2,:) = Y1(1:len);
coordinates_of_patches(3,:) = X2(1:len);
coordinates_of_patches(4,:) = Y2(1:len);

end