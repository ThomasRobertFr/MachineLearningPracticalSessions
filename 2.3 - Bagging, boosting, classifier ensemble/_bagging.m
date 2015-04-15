%% Clown data

clear all

nX=200;
X = zeros(nX,2);

X(1:nX/2  ,:) = randn(nX/2,2) + repmat([0 6], nX/2, 1);
X(nX/2+1:nX,1) = (rand(nX/2,1) - 0.5) * 6;
X(nX/2+1:nX,2) = X(nX/2+1:nX,1).^2 + 0.7*randn(nX/2,1);

Y = [ones(nX/2,1) ; zeros(nX/2,1)];

% découpage en apprentissage et test
[X, Y, Xtest, Ytest] = splitdata(X, Y, 0.3);
nX=length(X);
nXtest=length(Xtest);
plot(X(Y==1,1), X(Y==1,2), '.r'); hold on;
plot(X(Y==0,1), X(Y==0,2), '.b');
plot(Xtest(Ytest==1,1), Xtest(Ytest==1,2), 'or'); hold on;
plot(Xtest(Ytest==0,1), Xtest(Ytest==0,2), 'ob');
title('Jeu de test');

% Bagging

disp('K plus proches voisins');

[tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
    baggingEval(X, Y, Xtest, Ytest, 49, @knnc, @knn_map)

disp('Arbre de decision');

[tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
    baggingEval(X, Y, Xtest, Ytest, 49, @treec, @tree_map)

%% Jeu de données pima

data = csvread('pima-indians-diabetes.data');

X = data(:,1:end-1);
Y = data(:,end);

% découpage en apprentissage et test
[Xapp, Yapp, Xtest, Ytest] = splitdata(X, Y, 0.3);

%%
disp('K plus proches voisins');

[tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
    baggingEval(X, Y, Xtest, Ytest, 49, @knnc, @knn_map)
%%
disp('Arbre de decision');

[tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
    baggingEval(X, Y, Xtest, Ytest, 49, @treec, @tree_map)


%%

dataApp = tblread('sat.trn');
dataTest = tblread('sat.tst');

X = dataApp(:,1:end-1);
Y = dataApp(:,end);
Xtest = dataTest(:,1:end-1);
Ytest = dataTest(:,end);

%%
disp('K plus proches voisins');

[tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
    baggingEval(X, Y, Xtest, Ytest, 49, @knnc, @knn_map)
%%
disp('Arbre de decision');

[tauxErreurOOBMoy, ecartErreurOOB, tauxErreurTestBagging, tauxErreurTestSansBagging] = ...
    baggingEval(X, Y, Xtest, Ytest, 49, @treec, @tree_map)


%%







