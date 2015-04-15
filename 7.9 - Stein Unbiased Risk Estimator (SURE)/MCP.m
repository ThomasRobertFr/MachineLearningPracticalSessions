function x = MCP(x, g, l)

inds1 = x <= l;
inds2 = l < x & x <= g * l;

x(inds1) = 0;
x(inds2) = g / (g - 1) * (x(inds2) - l);

