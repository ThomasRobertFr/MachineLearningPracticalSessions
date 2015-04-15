function [tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
       baggingEval(X, Y, Xtest, Ytest, B, classifieur, estimateur)
   
nX = length(X);
nXtest = length(Xtest);

% BAGGING
tauxErreurs = zeros(B,1);
classifieurs = cell(1,B);
for i = 1:B
    % Tirage des jeux
    [bag, obag] = tireBootstrap(nX, nX);
    Xapp = X(bag, :);
    Yapp = Y(bag);
    Xval = X(obag, :);
    Yval = Y(obag);
    
    % Classification
    classifieurs{i} = classifieur(Xapp, ones(size(Yapp)), Yapp);
    probas = estimateur(classifieurs{i}, Xval, ones(size(Yval)), Yval);
    [~, classesVal] = max(probas, [], 2);
    tauxErreurs(i) = sum(Yval ~= classesVal)/length(classesVal)*100;
end
tauxErreurOOBMoy = mean(tauxErreurs); 
ecartErreurOOB = std(tauxErreurs); 

% ENSEMBLE DE CLASSIFIEURS
classes = zeros(nXtest, B);
nbClasses = 0;
for i = 1:B
    probas = estimateur(classifieurs{i}, Xtest, ones(size(Ytest)), Ytest);
    [~, classesTest] = max(probas, [], 2);
    nbClasses = size(probas, 2);
    classes(:,i) = classesTest;
end

% vote majorité
%Yclassif = mode(classes,2);

% vote pondéré
yHatsPond = zeros(nXtest, nbClasses);
for i = 1:B
    inds = sub2ind(size(yHatsPond), (1:nXtest)', classes(:,i));
    yHatsPond(inds) = yHatsPond(inds) + (100 - tauxErreurs(i));
end
[~, Yclassif] = max(yHatsPond, [], 2);

tauxErreurTestBagging = sum(Ytest ~= Yclassif)/nXtest*100;

% SANS BAGGING

classifieurSeul = classifieur(X, ones(size(Y)), Y);
probas = estimateur(classifieurSeul, Xtest, ones(size(Ytest)), Ytest);
[~, classesSeul] = max(probas, [], 2);
tauxErreurTestSansBagging = sum(Ytest ~= classesSeul)/length(classesSeul)*100;


