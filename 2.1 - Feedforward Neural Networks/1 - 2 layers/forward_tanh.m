function O = forward_tanh(W, I)
%Forward d'une couche de MLP
%avec une fonction de transfert tanh
%- W parametres de la couche : matrice (n entrees+1)*(n sorties)
%- I entrees de la couche : matrice n exemples*n entrees
%- O sorties de la couche : matrice n exemples*n sorties
%Forme lineaire
%On rajoute 1 à l’entrée pour faire le biais
nI=size (I, 1);
S = [I ones(nI, 1)] * W;
%Fonction de transfert
O = tanh(S);