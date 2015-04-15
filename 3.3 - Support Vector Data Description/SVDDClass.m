function [Xsup, alpha, b] = SVDDClass(Xi, C, kernel, kerneloption, options)

n = size(Xi, 1);

% compute the kernel
if (strcmp(kernel, 'polynomial'))
    G = (Xi*Xi' + ones(n) ) .^kerneloption;
else
    G = svmkernel(Xi, kernel, kerneloption);
end

% create usefull vectors
nx = diag(G);
e = ones(n,1);

% compute the solution
l = sqrt(eps);
[alpha, b, pos] = monqp(2*G,nx,e,1 ,C,l, 0) ;

% X support
Xsup = Xi(pos,:);
