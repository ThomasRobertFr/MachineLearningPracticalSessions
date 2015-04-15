function [C, O] = mlp2c_forward(W1, W2, X)
% calcule la sortie d'un perceptron multicouche de 2 couches avec une
% fonction de transfert tanh défini par ces matrices de paramètres W1 et
% W2

C = forward_tanh(W1, X);
O = forward_tanh(W2, C);