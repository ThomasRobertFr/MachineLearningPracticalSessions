function [tauxErreurAppMoy, ecartErreurApp, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
       randomSubspaceEval(X, Y, Xtest, Ytest, B, p, classifieur, estimateur)

nXtest = length(Xtest);

% RSM
tauxErreursApp = zeros(B,1);
classifieurs = cell(1,B);
ps = cell(1,B);
for i = 1:B
    % Tirage des jeux
    ps{i} = randperm(size(X,2), p);
    Xapp = X(:, ps{i});
    Yapp = Y;
    
    % Classification
    classifieurs{i} = classifieur(Xapp, ones(size(Yapp)), Yapp);
    probas = estimateur(classifieurs{i}, Xapp, ones(size(Yapp)), Yapp);
    [~, classesApp] = max(probas, [], 2);
    tauxErreursApp(i) = sum(Yapp ~= classesApp)/length(classesApp)*100;
end
tauxErreurAppMoy = mean(tauxErreursApp); 
ecartErreurApp = std(tauxErreursApp); 

% ENSEMBLE DE CLASSIFIEURS
classes = zeros(nXtest, B);
nbClasses = 0;
for i = 1:B
    probas = estimateur(classifieurs{i}, Xtest(:, ps{i}), ones(size(Ytest)), Ytest);
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
    yHatsPond(inds) = yHatsPond(inds) + (100 - tauxErreursApp(i));
end
[~, Yclassif] = max(yHatsPond, [], 2);

tauxErreurTestBagging = sum(Ytest ~= Yclassif)/nXtest*100;

% SANS RSM

classifieurSeul = classifieur(X, ones(size(Y)), Y);
probas = estimateur(classifieurSeul, Xtest, ones(size(Ytest)), Ytest);
[~, classesSeul] = max(probas, [], 2);
tauxErreurTestSansBagging = sum(Ytest ~= classesSeul)/length(classesSeul)*100;


