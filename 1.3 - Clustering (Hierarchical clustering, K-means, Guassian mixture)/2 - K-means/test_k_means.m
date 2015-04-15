function test_k_means(X, Klist, nbIte, param, figClusters, figJw)
    subI = length(Klist);
    subJ = nbIte + 1;
    
    % pour chaque k
    for i = 1:length(Klist)
        K = Klist(i);
        listes = zeros(size(X,1), nbIte);
        
        % pour chaque itération
        for j = 1:nbIte
        
            % on initialise les K centres
            C0 = init_centres(X, K);
            % on calcule nos clusters
            [C, liste, Jw] = k_moyennes(X, C0, param);
            listes(:,j) = liste;
            
            % affichage des clusters et des centres initiaux
            figure(figClusters);
            subplot(subI, subJ, (i - 1)*subJ + j);
            show_clusters(X, liste);
            title(['K = ' num2str(K) ' / Jw = ' num2str(Jw(end))]);
            hold on
            scatter(C(:,1),C(:,2), 50, 'k', 'd', 'fill');
            scatter(C0(:,1),C0(:,2), 50, 'k', 'd');
            
            % affichage de l'évolution de Jw
            figure(figJw);
            subplot(subI, subJ - 1, (i-1)*(subJ-1) + j);
            plot(Jw, '-*');
            title(['K = ' num2str(K)]);
        end
        
        % calcul des formes fortes à partir des intérations
        newlist = formes_fortes_from_listes(listes);
        
        % affichage formes fortes
        figure(figClusters);
        subplot(subI, subJ, i*subJ);
        show_clusters(X, newlist);
        title(['Formes fortes / Jw =' num2str(cout(X, newlist))]);
    end
end