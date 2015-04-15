function C = init_centres(X, K)
    
    d = size(X, 2);

    % points centrés entre 0 et 1
    C = rand(K, d) - repmat(0.5, K, d);
    
    % etendue des données
    etendue = max(X) - min(X);
    centre = (max(X) + min(X)) / 2;
    
    % multiplication par l'étendue
    C = C .* repmat(etendue, K, 1);
    
    % centrage de C sur X
    C = C + repmat(centre, K, 1);
end