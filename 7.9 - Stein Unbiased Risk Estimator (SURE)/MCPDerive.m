function x = MCPDerive(x, g, l)

inds1 = x <= l;
inds2 = l < x & x <= g * l;
inds3 = g * l < x;

x(inds1) = 0;
x(inds2) = g / (g - 1);
x(inds3) = 1;

