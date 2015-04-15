function [W1 W2] = mlp2c_init(m, n, o)
% Initialise les poids d'un perceptron multicouche à m neurones en entrée,
% n neurones dans la couche cachée, o neurones en sortie

W1 = ones(m, n);
W2 = ones(n, o);

