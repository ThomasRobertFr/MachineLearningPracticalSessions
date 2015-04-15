function C = nouveaux_centres(X, liste)
    K = max(liste);
    d = size(X, 2);
    
    C = zeros(K, d);
    
    % pour chaque cluster
    for k = 1:K
        % moyenne des points dans le cluster
        C(k, :) = mean(X(liste == k, :));
    end
end