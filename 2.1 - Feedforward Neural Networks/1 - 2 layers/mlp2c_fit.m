function [W1, W2, errorL] = mlp2c_fit(W1, W2, X, Y, nIter, pas, eps)
% qui estime les paramètres de ce perceptron multicouche à partir de
% l'ensemble (X, Y) pour un critère de type erreur quadratique. On
% utilisera la mise à jour des paramètres en bloc avec la taille du bloc
% égale au nombre d'exemples.

W1_old = W1*Inf;
W2_old = W2*Inf;

i = 1;
for j = 1:3
% while i < nIter & norm(W1 - W1_old) < eps &  & norm(W2 - W2_old) < eps
    % calcul des valeurs des unités (forward)
    [C, O] = mlp2c_forward(W1, W2, X);
    
    % gradient en sortie
    [errorL, gradEY] = criterion_mse(O, Y);
    
    % calcul du gradient (backward)
    [gradW1, gradW2] = mlp2c_backward(W1, W2, X, C, O, gradEY);
    
    % mise à jour de W1 et W2
    W1_old = W1;
    W2_old = W2;
    W1 = W1 + pas*gradW1;
    W2 = W2 + pas*gradW2;
    
    i = i + 1;
    
end