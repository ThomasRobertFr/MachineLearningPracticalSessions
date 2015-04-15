%% TP10

close all
clear all
clc

%% Régression logistique binomiale
%
%% Fonctions codées
% 
% On code plusieurs fonctions permettant de faire une régression linéraire.

%% Application aux données clowns
%
% Après avoir séparé les données en un ensemble de test et un ensemble
% d'apprentissage (80% en test / 20% en apprentissage), on applique les
% fonctions vues ci-dessus.

load clownsv7.mat

% Labels réencodés de -1 / 1 à 0 / 1
z = EncoderLabel01(y);

% Partage des données en deux : Apprentissage et Test
[xapp, zapp, xtest, ztest] = splitdata(X, z, 0.2);
n = size(X, 1);
napp = size(xapp, 1);
ntest = size(xtest, 1);

%% Frontière de décision linéaire 
% 
% On calcul d'abord la régression sur une matrice $\phi = [\mathbb{1} x_1
% x_2]$ afin d'obtenir une frontière de décision linéaire.
%
% On obtient alors une erreur d'environ 18% avec cette frontière linéaire.
% Cette erreur est raisonnable mais reste assez importante, d'autant qu'on
% pourrait sans doute beaucoup l'améliorer en utilisant une frontière de
% décision quadratique plus complexe et plus adaptée aux données.
%  
% Lorsque l'on cherche à classifier les données de test à partir des
% paramètres de décisions obtenus avec les données d'apprentissage, on se
% rend compte que la différence d'erreurs est assez faible entre les
% données d'apprentissage et de tests.

% matrices de données
phiApp = [ones(napp,1) xapp];
phiTest = [ones(ntest,1) xtest];

% application de la regression logistique
[theta,L] = ma_reg_log(phiApp, zapp);

% calcul des classes
zappExp = round(probaAPosteriori(theta, phiApp));
ztestExp = round(probaAPosteriori(theta, phiTest));

% erreur 
errApp = sum(zappExp ~= zapp)/napp;
errTest = sum(ztestExp ~= ztest)/ntest;

fprintf('Erreur en apprentissage : %i %%\n', round(errApp*100));
fprintf('Erreur en test : %i %%\n', round(errTest*100));

% calcul de la frontière de décision
xFront = [min(X(:,1)) max(X(:,1))]';
yFront = -(theta(1) + theta(2)*xFront)/theta(3);

% affichage
figure;
plot(xapp(zapp == 1,1),xapp(zapp == 1,2),'xr'); hold on;
plot(xapp(zapp == 0,1),xapp(zapp == 0,2),'xb');
plot(xtest(ztest == 1,1),xtest(ztest == 1,2),'.r');
plot(xtest(ztest == 0,1),xtest(ztest == 0,2),'.b');
plot(xFront, yFront, '-g');

legend('Apprentissage', 'Apprentissage', 'Test', 'Test', 'Frontière', 'Location', 'Best');

% evolution critere de convergence
figure;
plot(L);

%% Frontière de décision quadratique
% 
% On réalise donc une fronière de décision quadratique afin d'améliorer la
% performance de l'algorithme.
%
% On calcule pour cela la régression logistique avec la matrice $\phi = [1
% ~ x_1 ~ x_2 ~ x_1 x_2 ~ x_1^2 ~ x_2^2]$.
%
% Le reste fonctionne de la même façon que pour une frontière linéaire, on
% a simplement changé d'espace. La seule différence est que la frontière
% est quadratique, donc à tracer avec la fonction |contour| de Matlab.
%
% L'erreur tombe à 10%, soit presque la moitié de l'erreur avec une
% frontière linéaire. Cette erreur reste assez important car les données
% sont très mélangées autour de la frontière, il semble donc difficile de
% mieux séparer les données le plus excentrées des milieux de classes.
%
% La meilleure solution pour diminuer le taux d'erreur serait de rejeter
% les points incertains trop proche de la frontière de décision. Cependant,
% l'inconvénient de c

% matrices de données
phiApp = [ones(napp,1) xapp xapp(:,1).*xapp(:,2) xapp.^2];
phiTest = [ones(ntest,1) xtest xtest(:,1).*xtest(:,2) xtest.^2];

% application de la regression logistique
[theta,L] = ma_reg_log(phiApp, zapp);

% calcul des classes
zappExp = round(probaAPosteriori(theta, phiApp));
ztestExp = round(probaAPosteriori(theta, phiTest));

% erreur 
errApp = sum(zappExp ~= zapp)/napp;
errTest = sum(ztestExp ~= ztest)/ntest;

fprintf('Erreur en apprentissage : %i %%\n', round(errApp*100));
fprintf('Erreur en test : %i %%\n', round(errTest*100));

% calcul de la frontière de décision
[xx yy] = meshgrid(-4:0.1:4,-4:0.1:4);
zz = theta(1) + theta(2)*xx + theta(3)*yy + theta(4)*(xx.*yy) + ...
    theta(5) * (xx.^2) + theta(6) * (yy.^2);

% affichage
figure;
plot(xapp(zapp == 1,1),xapp(zapp == 1,2),'xr'); hold on;
plot(xapp(zapp == 0,1),xapp(zapp == 0,2),'xb');
plot(xtest(ztest == 1,1),xtest(ztest == 1,2),'.r');
plot(xtest(ztest == 0,1),xtest(ztest == 0,2),'.b');
contour(xx, yy, zz, [0 0]);

legend('Apprentissage', 'Apprentissage', 'Test', 'Test', 'Frontière', 'Location', 'Best');

% evolution critere de convergence
figure;
plot(L);

%% Régression logistique multimodale
% 
% Pour adapter le programme à 3 classes, il faudrait répéter l'opération de
% manière à séparer chaque groupe les uns des autres. Donc, pour l'ajout
% d'un (k+1)ième groupe, on aura k frontières de plus à celles déjà
% existantes.
% 
% En notant $\Theta = [\theta_1 ~ \theta_2 ~ ... ~ \theta_{k}]$ la matrice
% avec chaque paramètre de frontière en colonne, on peut simplement
% calculer les probabilités à posteriori. On aura dans la matrice de
% probabilités $P$ la probabilité pour chaque frontière de décision dans
% chaque colonne.
%
% La fonction de calcul de probabilité à posteriori est la suivant :
%
% [code]
%
% Le calcul de la fonction de régression logistique multimodale est plus
% compliquée. Nous n'avons pas réussi à faire fonctionner cette fonction
% correctement.
%
% La méthode que nous avons appliqué était de calculer $W$, $r$ et
% $\theta_i$ pour chaque colonne de la matrice $\Theta$. Malheureusement,
% cette méthode ne semble pas fonctionner correctement.