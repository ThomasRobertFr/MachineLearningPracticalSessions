function forest = randomForestTrain(X, Y, ...
                    hauteurMax, pureteSeuil, ...
                    nbArbres, nbFeaturesParArbre)

% taille donnees
[n,p] = size(X);

forest = cell(nbArbres, 1);

for i = 1:nbArbres
    
    % features de l'arbre i
    forest{i}.features = randperm(p, nbFeaturesParArbre);
    
    % tirage du bagging
    [bag, obag] = tireBootstrap(n, n);
    
    % arbre i
    forest{i}.tree = decisionTreeTrain(...
                        X(bag, forest{i}.features), Y(bag), ...
                        hauteurMax, pureteSeuil);
    
end