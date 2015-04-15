function tree = decisionTreeTrain(X, Y, hauteurRestante, pureteSeuil)

% taille
[n, p] = size(X);

% création d'un noeud
tree = struct();

% décision
tree.decision = souchebinairetrain(X, Y, ones(n,1));

% calcul des résultats dans les fils
Yhat = souchebinaireval(tree.decision, X);

% si des exemples dans le fils gauche ("-1") sont mal classés
Xgauche = X(Yhat == -1, :);
Ygauche = Y(Yhat == -1);
pureteG = abs(sum(Ygauche == 1) - sum(Ygauche == -1))/length(Ygauche);

if (pureteG < pureteSeuil && hauteurRestante > 0)
    tree.filsGauche = decisionTreeTrain(Xgauche, Ygauche, hauteurRestante-1, pureteSeuil);
end

% si des exemples dans le fils droit ("1") sont mal classés
Xdroit = X(Yhat == 1, :);
Ydroit = Y(Yhat == 1);
pureteD = abs(sum(Ydroit == 1) - sum(Ydroit == -1))/length(Ydroit);

if (pureteD < pureteSeuil && hauteurRestante > 0)
    tree.filsDroit = decisionTreeTrain(Xdroit, Ydroit, hauteurRestante-1, pureteSeuil);
end