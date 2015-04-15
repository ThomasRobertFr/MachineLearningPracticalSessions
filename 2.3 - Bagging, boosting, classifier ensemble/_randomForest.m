%% Clown data

clear all

nX=2000;
X = zeros(nX,2);

X(1:nX/2  ,:) = randn(nX/2,2) + repmat([0 6], nX/2, 1);
X(nX/2+1:nX,1) = (rand(nX/2,1) - 0.5) * 6;
X(nX/2+1:nX,2) = X(nX/2+1:nX,1).^2 + 0.7*randn(nX/2,1);

Y = [ones(nX/2,1) ; -ones(nX/2,1)];

% découpage en apprentissage et test
[X, Y, Xtest, Ytest] = splitdata(X, Y, 0.3);
nX=length(X);
nXtest=length(Xtest);
plot(X(Y==1,1), X(Y==1,2), '.r'); hold on;
plot(X(Y==-1,1), X(Y==-1,2), '.b');
plot(Xtest(Ytest==1,1), Xtest(Ytest==1,2), 'or'); hold on;
plot(Xtest(Ytest==-1,1), Xtest(Ytest==-1,2), 'ob');
title('Jeu de test');

%%

hauteurVals = 1:2:20;
erreurs = zeros(length(hauteurVals), 2) ;

for i = 1:length(hauteurVals)
    
    hauteur = hauteurVals(i);
    [a,b] = randomForestEval(X, Y, Xtest, Ytest, hauteur, 0.95, 49, 2);
    erreurs(i,:) = [a b];
    
end

figure
plot(hauteurVals, erreurs);
legend('Apprentissage', 'Test');
title('Erreur en fonction de la hauteur (purete = 95%, 49 arbres)');

% choix = 8

%%

pureteVals = [0.2 0.4 0.5 0.6 0.7 0.75 0.8:0.03:0.98];
erreurs = zeros(length(pureteVals), 2) ;

for i = 1:length(pureteVals)
    
    purete = pureteVals(i);
    [a,b] = randomForestEval(X, Y, Xtest, Ytest, 8, purete, 49, 2);
    erreurs(i,:) = [a b];
    
end

figure
plot(pureteVals, erreurs);
legend('Apprentissage', 'Test');
title('Erreur en fonction de la purete (Hmax = 8, 49 arbres)');

% choix = 0.85

%%

nbArbresVals = [5:6:60 71:121];
erreurs = zeros(length(nbArbresVals), 4) ;

for i = 1:length(nbArbresVals)
    
    nbArbres = nbArbresVals(i);
    [a,b,c,d] = randomForestEval(X, Y, Xtest, Ytest, 8, 0.85, nbArbres, 2);
    erreurs(i,:) = [a b c d];
    
end

%%

figure
subplot(2,1,1);
plot(nbArbresVals, erreurs(:,1:2));
legend('Apprentissage', 'Test');
title('Erreur en fonction du nombre d''arbres (Hmax = 8, purete = 85%)');
ylabel('Erreur');

subplot(2,1,2);
plot(nbArbresVals, erreurs(:,3:4));
ylabel('Ecart-type');
xlabel('Nombre d''arbres');
legend('Apprentissage', 'Test');


% choix 35

%%

erreurs = zeros(2, 2) ;

for i = 1:2
    
    nbArbres = nbArbresVals(i);
    [a,b] = randomForestEval(X, Y, Xtest, Ytest, 8, 0.85, 121, i);
    erreurs(i,:) = [a b];
    
end

figure
plot(1:2, erreurs);
legend('Apprentissage', 'Test');
title('Erreur en fonction du nombre de dimension par arbre (Hmax = 8, purete = 0.85, 121 arbres)');

% 2 dimensions

%%

[erreurTrain,erreurTest] = randomForestEval(X, Y, Xtest, Ytest, 8, 0.85, 35, 2)

%% Jeu de données pima

data = csvread('pima-indians-diabetes.data');

X = data(:,1:end-1);
Y = data(:,end);
Y(Y==0) = -1;

% découpage en apprentissage et test
[X, Y, Xtest, Ytest] = splitdata(X, Y, 0.3);
p = size(X,2);

%%

hauteurVals = [2 3 4 5  7 9 12];
erreurs = zeros(length(hauteurVals), 2) ;

for i = 1:length(hauteurVals)
    
    hauteur = hauteurVals(i);
    [a,b] = randomForestEval(X, Y, Xtest, Ytest, hauteur, 0.7, 69, p, 10);
    erreurs(i,:) = [a b];
    
end

figure
plot(hauteurVals, erreurs);
legend('Apprentissage', 'Test');
title('Erreur en fonction de la hauteur (purete = 70%, 69 arbres)');

% choix = 8

%%

pureteVals = [0.2 0.3 0.4 0.5 0.6 0.68 0.75 0.82 0.85 0.88 0.92 0.96];
erreurs = zeros(length(pureteVals), 2) ;

for i = 1:length(pureteVals)
    
    purete = pureteVals(i);
    [a,b] = randomForestEval(X, Y, Xtest, Ytest, 25, purete, 69, p, 5);
    erreurs(i,:) = [a b];
    
end

figure
plot(pureteVals, erreurs);
legend('Apprentissage', 'Test');
title('Erreur en fonction de la purete (Hmax = 25, 69 arbres)');

%%

nbDimVals = [2 3 4 5 6 7 8]; 
erreurs = zeros(nbDimVals, 2) ;

for i = 1:length(nbDimVals)
    
    nbDim = nbDimVals(i);
    [a,b] = randomForestEval(X, Y, Xtest, Ytest, 9, 0.7, 201, nbDim, 10);
    erreurs(i,:) = [a b];
    
end

figure
plot(nbDimVals, erreurs);
legend('Apprentissage', 'Test');
title('Erreur en fonction du nombre de dimension par arbre (Hmax = 9, purete = 0.7, 201 arbres)');

%%
[erreurTrain,erreurTest] = randomForestEval(X, Y, Xtest, Ytest, 9, 0.70, 69, 7)






