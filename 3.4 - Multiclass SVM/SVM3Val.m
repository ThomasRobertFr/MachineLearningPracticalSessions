function [y_pred] = SVM3Val(Xtest, Xsup, alpha, b, kernel, kerneloption)

[n, p] = size(Xsup);
n23 = 2/3*n;

% split b
b1 = b(1);
b2 = b(2);
b3 = b(3);

% split alpha
al12 = alpha(1:n/3) ;
al13 = alpha(n/3+1:n23) ;
al21 = alpha(n23+1:n23+n/3) ;
al23 = alpha(n23+n/3+1:2*n23) ;
al31 = alpha(2*n23+1:2*n23+n/3) ;
al32 = alpha(2*n23+n/3+1:end) ;

% compute kernel
K = svmkernel(Xtest, kernel, kerneloption, Xsup);

K1 = K(: ,1:n/3) ;
K2 = K(: ,n/3+1:n23) ;
K3 = K(: ,n23+1:end) ;

% predict
ypred1 = K1*al12 + K1*al13 - K2*al21 - K3*al31 + b1;
ypred2 = K2*al21 + K2*al23 - K1*al12 - K3*al32 + b2;
ypred3 = K3*al31 + K3*al32 - K1*al13 - K2*al23 + b3;

[~, yc] = max([ypred1 , ypred2 , ypred3]') ;

y_pred = yc;
