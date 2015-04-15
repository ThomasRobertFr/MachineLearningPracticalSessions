function x = SCADderive(x, a, l)

inds1 = x <= l;
inds1b = l < x & x <= 2*l;
inds2 = 2*l < x & x <= a * l;
inds3 = a * l < x;

x(inds1) = 0;
x(inds1b) = 1;
x(inds3) = 1;
x(inds2) = (a - 1) / (a - 2);

