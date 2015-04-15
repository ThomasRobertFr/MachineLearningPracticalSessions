%% TP BILAN
% Rémi MUSSARD
% Thomas ROBERT

%% Jeu de données abalone

clear all
clc
close all

% chargement des données
load abalone.mat
% supression de la première variable
X = X(:, 2:end);
% taille des données
[n, p] = size(X);

%% SVM

% modification de Y pour SVM
Y(Y == 0) = -1;

% erreurs en fonction de C vals
errs = [];

% valeurs de C à tester
Cvals = [0.001 0.01 0.05 0.1 0.5 1 5 10 30 70 100 200 500 700 1000 2000 3000 4000 5000 6000 9000 13000 15000 23000 30000];

for C = Cvals

    % résolution minimisation SVM par CVX
    cvx_begin quiet
        variable w(p)
        variable b(1)
        variable ksi(n)

        minimize(1/2 * w'*w + C * sum(ksi) )

        subject to

            Y .* (X * w + b) >= 1 - ksi;
            ksi >= 0;

    cvx_end
    
    % reconstruction de y
    yReconst = monsvmval(X, w, b);
    
    % calcul du taux d'erreur
    errs = [errs sum(Y ~= yReconst)/n*100];
    
end

% meilleure valeur de C
[meilleureErreur, indMeilleurC] = min(errs);
meilleurC = Cvals(indMeilleurC);
fprintf('Meilleur taux d''erreur : %.2f %% avec C = %i\n', meilleureErreur, C);

figure();
semilogx(Cvals, errs); hold on;
plot(meilleurC, meilleureErreur, '+r');
text(meilleurC, meilleureErreur+1, [int2str(round(meilleureErreur)) ' % ; C = ' num2str(meilleurC)])
title('Taux d''erreur en fonction de C');

%% Regression logistique

% modification de Y pour regression log
Y(Y == -1) = 0;

% matrices de donnéees
phi = [ones(n,1) X];
phiQuad = [ones(n,1) X X.^2];

% application de la regression logistique
[theta,~] = ma_reg_log(phi, Y);
[thetaQuad,~] = ma_reg_log(phiQuad, Y);

% calcul des classes
yExp = round(probaAPosteriori(theta, phi));
yExpQuad = round(probaAPosteriori(thetaQuad, phiQuad));

% erreur 
err = sum(yExp ~= Y)/n;
errQuad = sum(yExpQuad ~= Y)/n;
fprintf('Erreurs avec fontière linéaire : %.2f %%\n', err*100);
fprintf('Erreurs avec fontière quadratique : %.2f %%\n', errQuad*100);

%% Affichages

% Chaque variable
figure();
for i = 1:p
    
    X1=X(Y==1,i);
    X0=X(Y==0,i);
    n0 = length(X0);
    n1 = length(X1);
    
    subplot(2, ceil(p/2), i);
    hold on
    plot(rand(n0,1),X0, '.r');
    plot(rand(n1,1),X1, '.b');
    title(['Variable ' int2str(i)]);
end

% ACP
[D, U, moy]=mypca(X);
P = U(:,1:2);
Ct = projpca(X, moy, P);

subplot(2, ceil(p/2), i+1);
plot(Ct(Y==1,1), Ct(Y==1,2), '.b'); hold on;
plot(Ct(Y==0,1), Ct(Y==0,2), '.r');
title('ACP');

%% Jeu de données spambase

% chargement des données

clear all
load spambase.mat
[n, p] = size(X);

%% SVM

% modification de y pour SVM
y(y == 0) = -1;

% erreurs en fonction de C vals
errs = [];

% valeurs de C à tester
Cvals = [0.001 0.01 0.1 1 5 10 30 100 1e3 1e4 1e5 1e6];

for C = Cvals

    % résolution minimisation SVM par CVX
    cvx_begin quiet
        variable w(p)
        variable b(1)
        variable ksi(n)

        minimize(1/2 * w'*w + C * sum(ksi) )

        subject to

            y .* (X * w + b) >= 1 - ksi;
            ksi >= 0;

    cvx_end
    
    % reconstruction de y
    yReconst = monsvmval(X, w, b);
    
    % calcul du taux d'erreur
    errs = [errs sum(y ~= yReconst)/n*100];
    
end

% meilleure valeur de C
[meilleureErreur, indMeilleurC] = min(errs);
meilleurC = Cvals(indMeilleurC);
fprintf('Meilleur taux d''erreur : %.2f %% avec C = %i\n', meilleureErreur, C);

figure();
semilogx(Cvals, errs); hold on;
plot(meilleurC, meilleureErreur, '+r');
text(meilleurC, meilleureErreur+1, [int2str(round(meilleureErreur)) ' % ; C = ' num2str(meilleurC)])
title('Taux d''erreur en fonction de C');

%% Regression logistique

% modification de Y pour regression log
y(y == -1) = 0;

% matrices de donnéees
phi = [ones(n,1) X];
phiQuad = [ones(n,1) X X.^2];

% application de la regression logistique
[theta    ,~] = ma_reg_log_2(phi    , y);
[thetaQuad,~] = ma_reg_log_2(phiQuad, y);

% calcul des classes
yExp     = round(probaAPosteriori(theta    , phi    ));
yExpQuad = round(probaAPosteriori(thetaQuad, phiQuad));

% erreur 
err     = sum(yExp     ~= y)/n;
errQuad = sum(yExpQuad ~= y)/n;
fprintf('Erreurs avec fontière linéaire : %.2d %%\n', err*100);
fprintf('Erreurs avec fontière quadratique : %.2f %%\n', errQuad*100);

%% Regression logistique avec test et apprentissage

% modification de Y pour regression log
y(y == -1) = 0;

% 25% en apprentissage, 75% en test
[xApp, yApp, xTest, yTest] = splitdata(X, y, 0.25);
nApp = length(xApp);
nTest = length(xTest);

% matrices de donnéees
phiApp  = [ones(nApp,1) xApp  xApp.^2];
phiTest = [ones(nTest,1) xTest xTest.^2];

% application de la regression logistique
[theta,~] = ma_reg_log_2(phiApp, yApp);

% calcul des classes
yAppExp  = round(probaAPosteriori(theta, phiApp));
yTestExp = round(probaAPosteriori(theta, phiTest));

% erreur 
errApp = sum(yAppExp ~= yApp)/n;
errTest = sum(yTestExp ~= yTest)/n;
fprintf('Erreurs en apprentissage : %.2d %%\n', errApp*100);
fprintf('Erreurs en test : %.2f %%\n', errTest*100);

