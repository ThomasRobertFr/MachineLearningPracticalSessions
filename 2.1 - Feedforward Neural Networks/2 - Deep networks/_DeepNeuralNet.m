%% Codage et test des fonctions
%
% Afin de vérifier le fonctionnement de la toolbox, j'ai testé les diverses
% fonctions avec deux problèmes très simple : un problème de régression de
% la fonction $y=2\times x$, et un problème de classification de dimension
% 1.
%
% Le réseau testé est un réseau MLP avec une couche cachée, 1 entrée et 1
% ou 2 sorties selon les cas.
%
% En utilisant la toolbox, on se rend compte que le choix des fonctions
% d'activiation à un impact fort sur les perfomances du modèle. Par
% ailleurs, le fait que la méthode utilise un pas fixe et un nombre
% d'itération fixé fait que le choix de ces paramètres est très important :
% un pas trop faible rend la convergence très lente, un pas trop grand fait
% diverger au lieu de converger.
%
% Une bonne amélioration serait de fixer un critère de fin en plus d'un
% nombre d'itération maxmimum, et d'utiliser une méthode à pas variable
% pour accélérer la convergence.
%
% Notons également que pour augmenter légèrement la rapidité des calculs et
% surtout le confort d'utilisation, j'ai ajouté un paramètre à la fonction
% |onlinegrad| qui possède désormais une option |verbose| indiquant si on doit ou non
% afficher toutes les itérations.

% Exemple de problème de régression

clear all

net=ASINETfactory(1,[3 1],{'linear','linear'});

X = [1 2 3 4 5 6]';
Y = [2 4 6 8 10 12]';

[netout,learningErr,valError]=ASINETonlinegrad(net,X,Y,0.01,100,'mse', false);
YE=ASINETforward(netout,X);

%YE =
%    2.0510
%    4.0419
%    6.0328
%    8.0237
%   10.0146
%   12.0055

%%

% Exemple de problème de classification

clear all

net=ASINETfactory(1,[5 2],{'tanh','softmax'});

X = [1 2 3 7 8 9]';
Y = [0 0 0 1 1 1; 1 1 1 0 0 0]';

[netout,learningErr,valError]=ASINETonlinegrad(net,X,Y,0.001,1000,'nll', false);
YE=ASINETforward(netout,X)

%YE =
%    0.0095    0.9905
%    0.0185    0.9815
%    0.0518    0.9482
%    0.9666    0.0334
%    0.9761    0.0239
%    0.9791    0.0209

%% Générer le jeu de test
% 
% On génère un jeu de données que l'on découpe en apprentissage (10%) et en
% validation (90%).

clear all

nX=1000;
X = zeros(nX,2);

X(1:nX/2  ,:) = randn(nX/2,2) + repmat([0 6], nX/2, 1);
X(nX/2+1:nX,1) = (rand(nX/2,1) - 0.5) * 6;
X(nX/2+1:nX,2) = X(nX/2+1:nX,1).^2 + 0.7*randn(nX/2,1);

Y = [ones(nX/2,1) ; zeros(nX/2,1)];

% découpage en apprentissage et test
[Xapp, Yapp, Xtest, Ytest] = splitdata(X, Y, 0.1);
plot(X(Y==1,1), X(Y==1,2), '.r'); hold on;
plot(X(Y==0,1), X(Y==0,2), '.b');
plot(Xapp(Yapp==1,1), Xapp(Yapp==1,2), 'or'); hold on;
plot(Xapp(Yapp==0,1), Xapp(Yapp==0,2), 'ob');
title('Jeu de test');

%% Création d'un réseau
% 
% On se propose de tester un réseau avec 2 fonctions d'activation |tanh|.
% Nous avons 2 entrées qui sont les 2 dimensions de chaque donnée $x$, et
% une sortie.
%
% On teste donc ce réseau avec entre 1 et 10 neurones dans la couche
% cachée.
% 
% Nontons qu'a partir des données initiales, il est important de construire
% le bon vecteur cible à passer à la toolbox. Dans le cas d'une sortie de
% |tanh|, la cible contient des 0 et des 1.
% 
% On notera que le choix d'un critère type MSE se justifie par le fait
% qu'il faut pouvoir dériver la fonction, mais la vraie mesure du taux de
% bonne classification consiste à arrondir la sortie afin d'obtenir des 0
% et des 1, et de comparer à la cible.

% valeurs à tester
n_vals = 1:10;

% Stockage pour affichage
errApp = zeros(length(n_vals),1);
errVal = zeros(length(n_vals),1);
nbErrApp = zeros(length(n_vals),1);
nbErrVal = zeros(length(n_vals),1);

% Pour chaque nb de neurones dans la couche cachée
for i = 1:length(n_vals)
    
    n = n_vals(i);

    net=ASINETfactory(2,[n 1],{'tanh','tanh'})
    [netout,learningErr,valError]=ASINETonlinegrad(net,Xapp,Yapp,0.01,250,'mse', true, Xtest, Ytest);
    YE=ASINETforward(netout,Xapp);
    YE2=ASINETforward(netout,Xtest);

    errApp(i) = learningErr(end);
    errVal(i) = valError(end);
    nbErrApp(i) = sum(round(YE) ~= Yapp);
    nbErrVal(i) = sum(round(YE2) ~= Ytest);
    
end

%%

figure;
subplot(1,2,1);
plot(n_vals, errApp, '*-r'); hold on
plot(n_vals, errVal, '*-b');
title('Valeur du critère en fin de calcul');
legend('Apprentissage', 'Validation');
subplot(1,2,2);
plot(n_vals, nbErrApp/length(Xapp), '*-r'); hold on
plot(n_vals, nbErrVal/length(Xtest), '*-b');
title('Taux de classification en erreur');
legend('Apprentissage', 'Validation');

%% Reconnaissance de caractères
%
% On veut faire de la reconnaissance des caractères du fichier |uspsasi|
% pour distinguer le 1 des 8.
%
% Le prétraitement consiste à extraire uniquement les lignes qui nous
% intéressent dans la matrice x, et à générer un vecteur y de -1 et de 1.
%
% On sépare ensuite l'ensemble en appretissage (20%) et test (80%).
%
% Sans pré-apprentissage, on obtient des résultats très satisfaisants
% puisqu'on obtient un taux d'erreur en validation de seulement 7%. On peut
% sans doute espérer de meilleurs résultats en réglant de façon plus
% précise le pas et le nombre d'itérations.
%
% On fait ensuite un pré-apprentissage du réseau. L'objectif est de
% stabiliser les calculs en ne partant pas d'un réseau aléatoire, ce qui
% permet normalement de partir plus proche d'un minimum local.
%
% On obtient légèrement meilleurs que sans pré-apprentissage : 5% de
% mauvaise classification. Par ailleurs, l'écart-type de l'erreur de
% validation est 2 fois plus grand sans pré-apprentissage qu'avec. On
% remarque donc que le pré-apprentissage du réseau permet de stabiliser la
% méthode et d'accélérer l'apprentissage réel.

clear all
load uspsasi

% on ne garde que les 1 et les 8
x = [x(y==1,:); x(y==8,:)];
y = [-ones(sum(y==1),1); ones(sum(y==8),1)];
p = size(x,2);

% Stockage pour affichage
errApp = zeros(5,1);
errVal = zeros(5,1);
nbErrApp = zeros(5,1);
nbErrVal = zeros(5,1);

for i=1:5
    % découpage des donnes
    [xapp, yapp, xtest, ytest] = splitdata(x, y, 0.2);
    
    % calcul du réseau
    net=ASINETfactory(p, [64 16 1],{'tanh','tanh','tanh'})
    [netout,learningErr,valError]=ASINETonlinegrad(net,xapp,yapp,0.05,50,'mse', true,xtest,ytest);
    YE=round(ASINETforward(netout,xapp));
    YE2=round(ASINETforward(netout,xtest));

    errApp(i) = learningErr(end);
    errVal(i) = valError(end);
    nbErrApp(i) = sum(YE ~= yapp);
    nbErrVal(i) = sum(YE2 ~= ytest);
end

figure;
subplot(1,2,1);
plot(1:5, errApp, '*-r'); hold on
plot(1:5, errVal, '*-b');
title('Valeur du critère en fin de calcul');
legend('Apprentissage', 'Validation', 'Location', 'best');
subplot(1,2,2);
plot(1:5, nbErrApp/length(xapp), '*-r'); hold on
plot(1:5, nbErrVal/length(xtest), '*-b');
title('Taux de classification en erreur');
legend('Apprentissage', 'Validation', 'Location', 'best');

std(errVal)

% Stockage pour affichage
errApp = zeros(5,1);
errVal = zeros(5,1);
nbErrApp = zeros(5,1);
nbErrVal = zeros(5,1);

for i=1:5
    % découpage des donnes
    [xapp, yapp, xtest, ytest] = splitdata(x, y, 0.2);
    
    % calcul du réseau
    net=ASINETfactory(p, [64 16 1],{'tanh','tanh','tanh'})
    
    % pré-apprentissage
    % pour chaque layer a pré-apprendre
    H = xapp;
    for k = 1:net.nLayers - 1
        % construction du réseau de pré-apprentissage
        nbInputs = size(net.weight{k},1)-1
        nbInside = size(net.weight{k},2)
        netApp = ASINETfactory(nbInputs, [nbInside nbInputs],{net.type{k},'linear'})
        [netApp,~,~]=ASINETonlinegrad(netApp,H,H,0.0001,150,'mse', true);
        
        % sauvegarde du résultat
        net.weight{k} = netApp.weight{1};
        
        % entrée du prochain pré-apprentissage
        [~, Hmat]=ASINETforward(netApp,H);
        H=Hmat{2};
    end

    [netout,learningErr,valError]=ASINETonlinegrad(net,xapp,yapp,0.05,50,'mse', true,xtest,ytest);
    YE=round(ASINETforward(netout,xapp));
    YE2=round(ASINETforward(netout,xtest));

    errApp(i) = learningErr(end);
    errVal(i) = valError(end);
    nbErrApp(i) = sum(YE ~= yapp);
    nbErrVal(i) = sum(YE2 ~= ytest);
end

figure;
subplot(1,2,1);
plot(1:5, errApp, '*-r'); hold on
plot(1:5, errVal, '*-b');
title('Valeur du critère en fin de calcul');
legend('Apprentissage', 'Validation', 'Location', 'best');
subplot(1,2,2);
plot(1:5, nbErrApp/length(xapp), '*-r'); hold on
plot(1:5, nbErrVal/length(xtest), '*-b');
title('Taux de classification en erreur');
legend('Apprentissage', 'Validation', 'Location', 'best');

std(errVal)














