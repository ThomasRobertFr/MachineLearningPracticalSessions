%% Jeu clown

clear all

nX=2000;
X = zeros(nX,2);

X(1:nX/2  ,:) = randn(nX/2,2) + repmat([0 6], nX/2, 1);
X(nX/2+1:nX,1) = (rand(nX/2,1) - 0.5) * 6;
X(nX/2+1:nX,2) = X(nX/2+1:nX,1).^2 + 0.7*randn(nX/2,1);

Y = [-ones(nX/2,1) ; ones(nX/2,1)];

% découpage en apprentissage et test
[Xapp, Yapp, Xtest, Ytest] = splitdata(X, Y, 0.1);
plot(X(Y==1,1), X(Y==1,2), '.r'); hold on;
plot(X(Y==-1,1), X(Y==-1,2), '.b');
plot(Xtest(Ytest==1,1), Xtest(Ytest==1,2), 'or'); hold on;
plot(Xtest(Ytest==-1,1), Xtest(Ytest==-1,2), 'ob');
title('Jeu de test');

AdaBoostEval(Xapp, Yapp, Xtest, Ytest, 300);

%% Jeu pima

clear all

data = csvread('pima-indians-diabetes.data');

X = data(:,1:end-1);
Y = data(:,end);
Y(Y==0) = -1;

% découpage en apprentissage et test
[Xapp, Yapp, Xtest, Ytest] = splitdata(X, Y, 0.3);

AdaBoostEval(Xapp, Yapp, Xtest, Ytest, 300);
