function [ypred] = SVDDVal(Xtest, Xsup, alpha, b, kernel, kerneloption)

[n, p] = size(Xtest);

if (strcmp(kernel, 'polynomial'))
    K = (Xtest*Xsup' + ones(n, length(Xsup))) .^kerneloption;
    N_K = (sum(Xtest.^2, 2) + ones(n,1) ) .^kerneloption;
    ypred = N_K - 2*K*alpha - b;
else
    K = svmkernel(Xtest, kernel, kerneloption, Xsup);
    ypred = 1 - 2*K*alpha - b;
end
