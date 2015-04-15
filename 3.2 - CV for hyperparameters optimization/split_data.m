function [Xa,ya,Xval,yval] = split_data(Xi,yi,percent)
n = length(yi) ;
np = randperm(n) ;
Xa = Xi(np(1:round(percent*n) ) ,:) ;
ya = yi(np(1:round(percent*n) ) ) ;
Xval = Xi(np(round(percent*n) +1:end) ,:) ;
yval = yi(np(round(percent*n) +1:end) );
