%% Générer le jeu de test

X = zeros(100,2);

X(1:50  ,:) = randn(50,2) + repmat([0 6], 50, 1);
X(51:100,1) = (rand(50,1) - 0.5) * 6;
X(51:100,2) = X(51:100,1).^2 + 0.3*randn(50,1);

Y = ones(100,1);
Y(51:end) = -Y(51:end);

plot(X(Y==1,1), X(Y==1,2), '*r'); hold on;
plot(X(Y==-1,1), X(Y==-1,2), '*b');