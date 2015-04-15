function [K, Kt, G, ypred, nerr] = trainKernel(Xapp, yapp, Xt, yt, kernel, kerneloption, C)

n = length(yapp);
nt = length(yt);

K=svmkernel(Xapp,kernel ,kerneloption) ;
G = (yapp*yapp') .*K;
lambda = 1e-12;
e = ones(n,1) ;
[alpha ,b,pos] = monqp(G,e,yapp ,0 ,C,lambda ,0) ;
Kt = svmkernel(Xt,kernel ,kerneloption ,Xapp(pos ,:) ) ;
ypred = Kt*(yapp(pos) .*alpha) + b;
nerr = 100*length(find(yt.*ypred <0) ) /(nt) ;