function [Err_rate , ConfMat ] = Error_count( yt , predict_label )
ConfMat = zeros(2 ,2) ;
ConfMat(1 ,1) = length(find((yt == 1) &(predict_label==1) ) );
ConfMat(1 ,2) = length(find((yt == 1) &(predict_label== -1) )) ;
ConfMat(2 ,1) = length(find((yt == -1) &(predict_label==1) )) ;
ConfMat(2 ,2) = length(find((yt == -1) &(predict_label== -1) ) ) ;
nt = length(yt) ;
Err_rate = 100* ( ConfMat(1 ,2) + ConfMat(2 ,1) ) / nt;