
% On fait le clustering de N=12 points en K=3 clusters. On répète l'expérience
% 3 fois et obtient le resultat dans clus. Chaque colonne de clus
% représente l'affectation des points pour chaque expérience.
N = 12;
clus =[

     1     2     3
     1     2     3
     1     2     3
     2     3     1
     2     3     1
     2     3     1
     3     1     2
     3     1     2
     3     1     2
     1     3     1
     1     3     1
     1     3     1];
 
 
 %%
 % Le but est de détecter les points qui sont tombés dans le méme cluster (peu
 % importe le numéro du cluster) sur les trois expériences. On remarque les
 % 3 premiers points sont dans le méme cluster les 3 fois. Ceci se
 % manifeste par le motif [1 2 3] dans la matrice clus. Les points 4 é 6
 % font de méme (motif [2 3 1]). Les 3 derniers points tombent aussi
 % ensemble é chaque fois (motif [1 3 1]). Détecter les formes formes
 % revient é identifier ces motifs
 
[tmp tmp2 listeformesfortes]=unique(clus,'rows') ;

%%
% former les clusters donnés par les formes fortes
Nff = max(listeformesfortes);
effectifs= zeros(Nff, 1);
newlist = zeros(N,1);
for c=1:Nff
    ind = find(listeformesfortes==c) ;    
    effectifs(c)=length(ind);
    newlist(ind)=c; 
end
% Nff : nombre de formes fortes
% newlist : contient les affectations des points dans les clusters trouvés
% avec les formes fortes
% effectifs : contient le nombre de points de chaque cluster identifié par
% les formes fortes
% on peut maintenant fourner K-means avec K = Nff
