function [err, gradEY] = criterion_mse(O, Y)
%Critere L en MSE
%- O sorties du MLP
%- TARGET valeur cible
%- err valeur de L
%- gradCrit Grad de L par rapport à O
err = sum((Y - O).^2);
gradEY = (Y - O);