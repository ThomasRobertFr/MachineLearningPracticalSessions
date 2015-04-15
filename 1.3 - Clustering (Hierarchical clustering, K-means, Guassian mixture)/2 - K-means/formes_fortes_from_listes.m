function newlist = formes_fortes_from_listes(listes)

    N = size(listes,1);

    [~, ~, listeformesfortes]=unique(listes,'rows') ;

    Nff = max(listeformesfortes);
    effectifs = zeros(Nff, 1);
    newlist = zeros(N,1);

    for c=1:Nff
        ind = find(listeformesfortes==c) ;    
        effectifs(c)=length(ind);
        newlist(ind)=c; 
    end

end