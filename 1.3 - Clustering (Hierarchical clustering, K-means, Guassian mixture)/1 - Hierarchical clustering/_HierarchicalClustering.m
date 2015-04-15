% TP3

clear all
close all

%% Calcul de distance
%
% <voir fonction distance.m>

%% Fonctionnement d'aggclust
% 
% aggclust crée une hiérachie ascendante des clusters. Il initialise le
% niveau 1 en créant 1 cluster par point. On boucle ensuite de 2 au nombre
% de points et en rassemblant à chaque fois les deux clusters les plus
% proches.

%% Fonction calc_dendro
% 
% Nous avons écrit une petite fonction qui se charge de calculer et
% d'afficher les deux dendrogrammes différents (un pour chaque méthode) à
% partir de données.


%% Classification ASI4
% 
% On voit que la méthode single n'arrive pas du tout à trouver de groupes
% alors que la méthode complete forme bien des groupes qui semblent (au vu
% du nombre d'étudiants par groupe) relativement bien répartis.

load('asi4.mat');
fig = figure();
calc_dendro(data, true);
set(fig, 'Position', [100 100 850 420]);

%% DS2
%
% Afin de pouvoir visualiser graphiquement les clusters, on écrit une
% fonction show_clusters(data, level, nbClust) permettant d'afficher
% nbClust clusters.
%
% On constate que cette méthode ne donne pas toujours les deux clusters
% que l'on attendrait (un par losange) mais regroupe parfois les pointes
% (un cluster avec les pointes inférieures et un avec les pointes
% supérieures).

load ds2.dat

data = mydownsampling(ds2, 30);

fig = figure();
[M, level, ~] = calc_dendro(data, true);
set(fig, 'Position', [100 100 850 420]);

% affichage

fig = figure();
for i=1:6
    subplot(2,3,i);
    show_clusters(data, level, i);
    title([int2str(i) ' clusters']);
end
set(fig, 'Position', [100 100 750 420]);

%% George
%
% Avec les données "george", les clusters semblent plus proche de ce que
% l'on attends (1 cluster par lettre lorsque l'on choisi 6 clusters), et le
% résultat semble relativement stable vis à vis du sous-échantillonage,
% contraitement à ce que l'on observait avec les données "ds2".

load george.dat

data = mydownsampling(george, 15);

fig = figure();
[M, level, ~] = calc_dendro(data, true);
set(fig, 'Position', [100 100 850 420]);

% affichage

fig = figure();
for i=1:6
    subplot(2,3,i);
    show_clusters(data, level, i);
    title([int2str(i) ' clusters']);
end
set(fig, 'Position', [100 100 980 420]);

