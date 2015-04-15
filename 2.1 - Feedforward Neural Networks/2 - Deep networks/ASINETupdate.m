function netout=ASINETupdate(netin,grad,rho)
%%
% Mise à jour des poids d'un réseau
%
% netin: réseau entrée
% grad:  gradient
% rho: pas d'apprentissage
%  
% netout: réseau sortie
%

netout=netin;

% On boucle sur chaque couche du réseau
for l=1:netin.nLayers
    netout.weight{l}=netin.weight{l}-rho*grad.weight{l}; 
end