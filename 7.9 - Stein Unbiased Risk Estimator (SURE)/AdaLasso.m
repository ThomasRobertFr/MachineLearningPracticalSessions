function x = AdaLasso(x, q, l)

xpower = x.^(1+q);

inds1 = xpower <= l;
inds2 = xpower > l;

x(inds1) = 0;
x(inds2) = x(inds2) - l ./ (x(inds2).^l);
x(x < 0) = 0;

