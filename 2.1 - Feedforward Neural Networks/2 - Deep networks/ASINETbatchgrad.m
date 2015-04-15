function [netout,learningErr,valError]=ASINETbatchgrad(netin,X,TARGET,rho,maxiter,criterion,Xval,TARGETval)
%%
% Apprentissage d'un PMC par retropropagation du gradient en batch
%
% netin: PMC en entrée
% X: ensemble d'apprentissage
% TARGET: cibles 
% rho: pas d'apprentissage
% criterion: critère d'apprentissage ('mse')
% Xval: ensemble de validation [optionnel]
% TARGETval: cibles de validation [optionnel]
%
% netout: PMC en sortie

disp('ONLINE GRADIENT');

if nargin==8
  valid=1;
else
  valid=0;
end

netout=netin;
nX = size(X,1);

%-------------------
% Calcul des erreurs
YE=ASINETforward(netout,X);
err=ASINETcriterion(TARGET,YE,criterion);
learningErr=err;
disp(['Learning error: ',num2str(err)]);

valError=[];
if (valid)
    YE=ASINETforward(netout,Xval);
    err=ASINETcriterion(TARGETval,YE,criterion);
    valError=err;
    disp(['Validation error: ',num2str(err)]);
end




%-------------------

% On boucle jusqu'au maximum d'itérations
for i=1:maxiter

    disp(['Iteration: ',num2str(i)])

    % On boucle sur chaque exemple
    netB = netout;
    for ix=1:nX
      [YE,state]=ASINETforward(netB,X(ix,:));
      %On calcule le gradient correspondant à l'exemple courrant
      [err, gradOut]=ASINETcriterion(TARGET(ix,:),YE,criterion);
      gradx=ASINETbackward(netB ,state,gradOut);
      %On met à jour le réseau
      netout=ASINETupdate(netout,gradx,rho);
    end


    % Calcul des errers
    YE=ASINETforward(netout,X);
    err=ASINETcriterion(TARGET,YE,criterion);
    learningErr=[learningErr err];
    disp(['Learning error: ',num2str(err)]);

    if (valid)
        YE=ASINETforward(netout,Xval);
        err=ASINETcriterion(TARGETval,YE,criterion);
        valError=[valError err];
        disp(['Validation error: ',num2str(err)]);
end

    disp('--------------');
end



