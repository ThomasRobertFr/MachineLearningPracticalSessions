%% Chargement des données

clear all
load iris

% split data
[x, y, xtest, ytest] = splitdata(x, y, 2/3);

% ajout d'outliers
xtest(end+1, :) = [15 5 5 5];
xtest(end+1, :) = [15 5 5 5];
xtest(end+1, :) = [15 5 5 5];
xtest(end+1, :) = [15 5 5 5];
ytest(end+1:end+4) = [0;0;0;0];

% data size
[n, p] = size(x); % taille du jeu
nClasses = max(y);

% création de la matrice de sortie du réseau
yn = y;
y = zeros(n, nClasses);
y(sub2ind(size(y), (1:n)', yn)) = 1;

%% Paramètres

Kvals = [3:10 12:2:20 25 30]; 
nIte = 20;

errorRate = zeros(1,length(Kvals));
results = {};

%% test des différentes valeurs de K

for kind = 1:length(Kvals)
    K = Kvals(kind); % nombre de classes pour k-means

    for ite = 1:nIte

        %% Apprentissage

        % topologie
        nIn = p;
        nRBF = K;
        nOut = nClasses;

        % classes kmeans
        classes = kmeans(x, K);

        % paramètres des gaussiennes
        mu = {};
        sigma = {};
        for i = 1:K
            mu{i} = mean(x(classes == i, :));
            sigma{i} = cov(x(classes == i, :)) + 0.01 * eye(p);
        end

        % Sortie du RBF
        xRBF = ones(n, K);
        for i = 1:K
            xRBF(:, i) = mvnpdf(x, mu{i}, sigma{i});
        end

        % Calcul des poids de la couche de sortie
        Wout = (xRBF'*xRBF)\(xRBF'*y);

        %% Test

        ntest = size(xtest, 1); % taille du jeu
        xRBFtest = ones(ntest, K);
        for i = 1:K
            xRBFtest(:, i) = mvnpdf(xtest, mu{i}, sigma{i});
        end

        % Calcul de la sortie
        yOut = exp( xRBFtest * Wout);
        yOut = yOut ./ repmat(sum(yOut, 2), 1, nOut);

        % Calcul de la prédiction
        RBFconfidence = max(xRBFtest, [], 2);
        [~, yHat] = max(yOut, [], 2);
        yHat(RBFconfidence < 0.01) = 0;
        results{kind} = yHat;

        % Taux d'erreur
        newError = sum(yHat ~= ytest)/length(ytest);
        errorRate(kind) = (errorRate(kind) * (ite - 1) + newError) / ite;
    end
    
end

%% Plot résultats

figure;
plot(Kvals, errorRate*100);
xlabel('K');
ylabel('Erreur (%)');
title('Erreur en test en fonction de K')
[~, bestKind] = min(errorRate);

ytestMat = zeros(ntest, nClasses + 1);
ytestMat(sub2ind(size(ytestMat), (1:ntest)', ytest + 1)) = 1;
yhatMat = zeros(ntest, nClasses + 1);
yhatMat(sub2ind(size(yhatMat), (1:ntest)', results{bestKind} + 1)) = 1;

figure;
plotconfusion(ytestMat', yhatMat');
