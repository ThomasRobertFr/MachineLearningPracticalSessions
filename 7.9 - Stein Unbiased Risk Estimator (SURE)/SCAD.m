function x = SCAD(x, a, l)

inds1 = x <= 2*l;
inds2 = 2*l < x & x <= a * l;

x(inds1) = max(0, x(inds1) - l);
x(inds2) = ((a - 1) * x(inds2) - a * l) / (a - 2);

