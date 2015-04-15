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
    
    % Création jeux
    dataApp = prdataset(Xapp, Yapp);
    dataVal = prdataset(Xval, Yval);
    
    % Classification
    classifieurs{i} = classifieur(dataApp);
    probas = estimateur(dataVal, classifieurs{i});
    [~, classesVal] = max(probas.data, [], 2);
    tauxErreurs(i) = sum(dataVal.nlab ~= classesVal)/length(classesVal)*100;
end
tauxErreurOOBMoy = mean(tauxErreurs); 
ecartErreurOOB = std(tauxErreurs); 

% ENSEMBLE DE CLASSIFIEURS
dataTest = prdataset(Xtest, Ytest);
target = dataTest.nlab;

classes = zeros(nXtest, B);
for i = 1:B
    probas = estimateur(dataTest, classifieurs{i});
    [~, classesTest] = max(probas.data, [], 2);
    classes(:,i) = classesTest;
end

Ytarget = dataTest.nlab;
Yclassif = mode(classes,2);

tauxErreurTestBagging = sum(Ytarget ~= Yclassif)/nXtest*100;

% SANS BAGGING

% Création du jeu
dataApp = prdataset(X, Y);
dataTest = prdataset(Xtest, Ytest);

% Classification
classifieurSeul = classifieur(dataApp);
probas = estimateur(dataTest, classifieurSeul);
[~, classesSeul] = max(probas.data, [], 2);
Ytarget = dataTest.nlab;
tauxErreurTestSansBagging = sum(Ytarget ~= classesSeul)/length(classesSeul)*100;
