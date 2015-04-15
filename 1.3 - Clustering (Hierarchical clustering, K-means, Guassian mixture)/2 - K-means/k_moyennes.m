function [C, liste, Jw] = k_moyennes(X, C0, param)
    C = C0;
    
    % init des couts
    JwPrec = 1;
    JwNew = 0;
    Jw = [];
    
    % tant que le cout varie de plus de 1e-3
    while abs(JwPrec - JwNew) > 1e-3
        JwPrec = JwNew;
        
        % calcul des distances
        M = distance(X, C, param);
        % affectation et couts
        [JwNew, liste] = affectation_cout(M);
        Jw = [Jw ; JwNew];
        % calcul des nouveaux centres
        C = nouveaux_centres(X, liste);
    end
end