function [netout,learningErr,valError]=ASINETonlinegrad(netin,X,TARGET,rho,maxiter,criterion,verbose,Xval,TARGETval)
%%
% Apprentissage d'un PMC par retropropagation du gradient en ligne
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

if (verbose)
    disp('ONLINE GRADIENT');
end

if nargin==9
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
if (verbose)
    disp(['Learning error: ',num2str(err)]);
end

valError=[];
if (valid)
    YE=ASINETforward(netout,Xval);
    err=ASINETcriterion(TARGETval,YE,criterion);
    valError=err;
    if (verbose)
        disp(['Validation error: ',num2str(err)]);
    end
end




%-------------------

% On boucle jusqu'au maximum d'itérations
for i=1:maxiter

    if (verbose)
        disp(['Iteration: ',num2str(i)]);
    end

    % On boucle sur chaque exemple
    for ix=1:nX
      [YE,state]=ASINETforward(netout,X(ix,:));
      %On calcule le gradient correspondant à l'exemple courrant
      [err, gradOut]=ASINETcriterion(TARGET(ix,:),YE,criterion);
      gradx=ASINETbackward(netout,state,gradOut);
      %On met à jour le réseau
      netout=ASINETupdate(netout,gradx,rho);
    end


    % Calcul des errers
    YE=ASINETforward(netout,X);
    err=ASINETcriterion(TARGET,YE,criterion);
    learningErr=[learningErr err];
    if (verbose)
        disp(['Learning error: ',num2str(err)]);
    end
    
    if (valid)
        YE=ASINETforward(netout,Xval);
        err=ASINETcriterion(TARGETval,YE,criterion);
        valError=[valError err];
        if (verbose)
            disp(['Validation error: ',num2str(err)]);
        end
    end

    if (verbose)
        disp('--------------');
    end
end



