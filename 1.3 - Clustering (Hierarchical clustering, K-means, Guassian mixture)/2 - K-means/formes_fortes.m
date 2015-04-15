function newlist = formes_fortes(X, nbIte, param, K)

    N = size(X,1);

    listes = zeros(N,nbIte);

    for j = 1:nbIte
        C0 = init_centres(X, K);
        [~, liste, ~] = k_moyennes(X, C0, param);
        listes(:,j) = liste;
    end
    
    newlist = formes_fortes_from_listes(listes);
    
end