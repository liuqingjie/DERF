function flt = gaussian_1d(sigma, mean, fsz)
sz = floor((fsz-1)/2);
v = sigma*sigma*2;
x=-sz:sz;
flt = exp( -((x-mean).^2)/v );
n=sum(flt);
flt=flt/n;
end