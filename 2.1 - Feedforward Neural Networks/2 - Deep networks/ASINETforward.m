function [YE, varargout]=ASINETforward(net,X)
%%
% Calcul la propagation sur un réseau pour plusieurs exemples
%
% net: réseau
% X: esemble d'exemples
%
% YE: estimation de la sortie 
% varagout{1}: estimation des valeurs de toutes les couches (état du réseau)

nX = size(X,1);
n0 = size(X,2) + 1;

% On propage l'information de couche en couche
state{1}=X;

for l=1:net.nLayers
    % Forme linéaire des entrées, on concaténe par 1 les exemples pour le biais
    L = [state{l} ones(nX,1)]*net.weight{l} ;
    % Fonction de transfert suivant le type de la couche
    switch lower(net.type{l})
    case 'linear'
      state{l+1} = L;
    case 'tanh'
      state{l+1} = tanh(L);
    case 'sigm'
      state{l+1} = 1./(1+exp(-L));
    case 'softmax'
      n0 = size(L,2);
      state{l+1} = exp(L)./repmat(sum(exp(L), 2), 1, n0);
    case 'logsoftmax'
      n0 = size(L,2);
      state{l+1} = L - repmat(log(sum(exp(L), 2)), 1, n0);
    otherwise
      error('Unknown transfert function');
    end
end

YE=state{l+1};

if nargout>1
 varargout{1}=state;
end