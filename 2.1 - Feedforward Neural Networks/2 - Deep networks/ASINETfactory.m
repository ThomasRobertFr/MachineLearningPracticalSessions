function net=ASINETfactory(ninputs,layerNN,layerType)
%%
% Construit un PMC 
%
% ninputs: nombre d'entrées
% layerNN: vecteur contenant le nombre de neurones pour chaque couche, ex: [4,5,2]
% layerType: structure contenant le  type de la fonction de transfert pour chaque couche, ex: {'sigm','linear','tanh'}
% 
% net: réseau construit
%

% On calcul le nombre de couche
net.nLayers=length(layerNN);

% On construit le reseau couche à couche
for l=1:net.nLayers
   net.type{l}=layerType{l};
   %Initialisation aléatoire des poids
   if l==1
      net.weight{1}=0.5*randn(ninputs+1,layerNN(1));
   else
      net.weight{l}=0.5*randn(layerNN(l-1)+1,layerNN(l));
   end
end