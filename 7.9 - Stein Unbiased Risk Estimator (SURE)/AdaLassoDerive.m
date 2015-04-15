function x = AdaLassoDerive(x, q, l)

xpower = x.^(1+q);

inds1 = xpower <= l;
inds2 = xpower > l;

x(inds1) = 0;
x(inds2) = 1 + l * q ./ xpower(inds2);

