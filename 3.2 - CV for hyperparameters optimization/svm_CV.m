function [b_o,C_o,Err]=svm_CV(Xa,ya,Xval,yval,b_grid ,C_grid ,kernel ,la)
na = length(ya) ;
e = ones(na,1) ;
for i = 1:length(b_grid)
    kerneloption = b_grid(i) ;
    K=svmkernel(Xa,kernel ,kerneloption) ;
    G = (ya*ya') .*K;
    for j=1:length(C_grid)
        [alpha ,b,pos] = monqp(G,e,ya,0 ,C_grid(j) ,la,0) ;
        Kval = svmkernel(Xval,kernel ,kerneloption ,Xa(pos ,:) ) ;
        predict_label = sign(Kval*(ya(pos) .*alpha) + b) ;
        [Err_rate , ConfMat ] = Error_count( yval , predict_label ) ;
        Err(i,j) = Err_rate;
    end
end
[Errmin C_ind] = min(min(Err) ) ;
C_o = C_grid(C_ind) ;
[Errmin b_ind] = min(min(Err') ) ;
b_o = b_grid(b_ind) ;
