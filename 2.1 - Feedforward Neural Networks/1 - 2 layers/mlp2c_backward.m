function [gradW1, gradW2] = mlp2c_backward(W1, W2, X, C, O, gradEY)
% calcule la mise à jour des paramètres de ce perceptron pour un SEUL
% exemple pour un gradient d'erreur en sortie gradEY

[gradW2, gradEC] = backward_tanh(W2, C, O, gradEY);
[gradW1, gradEX] = backward_tanh(W1, X, C, gradEC);

