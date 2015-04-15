%% Méthode d'évaluation
%
% Une composante importante du TP a consisté a travailler sur une fonction
% permettant d'évaluer les performances d'un classifieur en top i en
% calculant les taux de classification, rejet en ambiguité et confusion.
%
% Ces scores sont calculés à partir d'une matrice $X$ avec une ligne par
% exemple à classifier, et une colonne par classe. La valeur $X_{ij}$ de
% la matrice correspond à une mesure de la confiance du classifieur dans le
% fait que l'exemple $i$ est de la classe $j$.
%
% Cette méthode est donc généralisable et applicable pour évaluer n'importe
% quel résultat, que la matrice $X$ contienne des mesures, des rangs ou des
% votes.
%
% On considère qu'un exemple $i$ dont la classe réelle est $j$ est "classé"
% en top $k$ si $X_{ij}$ fait parti des $k$ plus fortes valeurs de la ligne
% $X_{i\bullet}$, et que la valeur $X_{ij}$ n'a pas de valeur égale en
% dehors des $k$ plus fortes valeurs de $X_{i\bullet}$, c'est à dire qu'il
% n'y a pas de conflit avec le score $X_{ij}$ en dehors du top $k$. Notons
% également que $X_{ij}$ doit être supérieur à 0, puisque les scores égaux
% à zéro correspondent aux cas non décidés par le classifieur.
%
% On considère qu'un exemple $i$ dont la classé réelle est $j$ est "rejeté
% en ambiguité" en top $k$ si $X_{ij}$ fait parti des $k$ plus fortes
% valeurs de la ligne $X_{i\bullet}$ mais que la valeur $X_{ij}$ a au moins
% une valeur égale en dehors des $k$ plus fortes valeurs de $X_{i\bullet}$,
% c'est à dire qu'il y a un conflit avec le score $X_{ij}$ en dehors
% du top $k$. L'exemple $i$ peut également être rejeté si aucun score de la
% ligne $X_{i\bullet}$ est supérieur à 0, c'est à dire que l'on rejette en
% ambiguité un exemple pour lequel le classifieur ne donne aucun résultat.
% Ce cas arrive en combinaison de mesure par produit par exemple.
%
% On considère qu'un exemple $i$ dont la classé réelle est $j$ est "confus"
% en top $k$ si $X_{ij}$ ne fait pas parti des $k$ plus fortes valeurs de
% la ligne $X_{i\bullet}$. C'est à dire les cas qui ne sont ni "classés",
% ni "rejetés".
%
%% Performance reco Top1 et Top5
%
% On mesure les performances des divers classifieurs en top 1 et top 5. On
% constate que les performances des classifieurs sont très variables,
% allant pour le top 1 et 60 à 90%.
%
% En toute logique, les performances en top 5 sont supérieures à celles en
% top 1, allant de 88 à 96%.

clear all
load data

results = zeros(nbCl, 10);
for i = 1:nbCl
    [results(i,1:2:9), results(i,2:2:10)] = evaluerPerfs(Xapp{i}, yapp{i}, 5);
end

showTable(results, {'Classif T1', 'Rejet T1', 'Classif T2', 'Rejet T2', 'Classif T3', 'Rejet T3', 'Classif T4', 'Rejet T4', 'Classif T5', 'Rejet T5'}, {'Cl1', 'Cl2', 'Cl3', 'Cl4', 'Cl5'})
figure;
subplot(1,2,1);
plotResults(results(:,1:2:9)');
subplot(1,2,2);
plotResults(results(:,2:2:10)', 'rejet');

%% Méthodes de combinaison de type "Classe"
%
% On trace pour chaque méthode (vote à la pluralité, la majorité, à la
% pluralité pondérée) et pour chaque jeu de données (apprentissage et test)
% les scores en classification, rejet d'ambiguité et confusion.
%
% De manière générale, les performances obtenues sont bonnes, bien
% meilleures que les performances des classifieurs pris séparément,
% montrant bien (au moins dans ce cas) l'apport des mélanges de
% classifieurs.
%
% On constate sans étonnement que le vote à la pluralité provoque moins de
% rejet que le vote à la majorité, puisque les résultats du vote à la
% majorité sont les mêmes que ceux du vote à la pluralité en rejetant les
% cas qui ne sont pas votés par au moins 50% des classifieurs.
% 
% Sur le même principe, il n'est pas étonnant de constater que le vote à la
% majorité à beaucoup moins de confusion que le vote à la pluralité,
% puisque les cas ambigus où les classifieurs sont très partagés seront
% rejetés en ambiguité.
%
% Enfin, le vote à la pluralité pondérée est celui qui offre les meilleurs
% résultats en classifications. Cependant, ceci peut être en partie
% expliqué par le fait que l'utilisation de pondération des votes fait
% qu'il n'y a aucun cas dans le jeu de données où il y a ambiguité. C'est
% donc la solution qui a le meilleur score en classification, mais qui a
% également le plus mauvais score (le plus fort) en confusion.
%
% Il est donc impossible de juger qu'une de ces solutions est meilleure que
% les autres puisque pour chaque, une augmentation des performances en
% classification entrainer une augmentation du taux de confusion. Le choix
% de la "meilleure" solution pour un cas donné pourrait être fait en
% attribuant des coûts aux 3 cas (classification, rejet, confusion) par
% exemple.

clear all
load data

lignes = {'Pluralité app', 'Pluralité test', 'Majorité app', 'Majorité test', 'Pondération app', 'Pondération test'};
colonnes = {'% Classification', '% Rejet', '% Confusion'};
results = zeros(6, 3);

[results(1,1), results(1,2), results(3,1), results(3,2), results(5,1), results(5,2)] = ...
    combinaisonClasse(Xapp, Xapp, yapp{1});
[results(2,1), results(2,2), results(4,1), results(4,2), results(6,1), results(6,2)] = ...
    combinaisonClasse(Xtest, Xapp, ytest{1});

results(:,3) = 1 - results(:,2) - results(:,1);

showTable(results*100, colonnes, lignes);

lignesVote = lignes;
resultsVote = results;
save('resultsVote', 'resultsVote', 'lignesVote');

%% Méthodes de combinaison de type "Rang"
%
% On essaye maintenant des méthodes de combinaison de type rang. On ne
% considère donc plus les probabilités en sortie des classifieurs mais
% simplement l'ordre de ces probabilités, correspondantes au rang de chaque
% prédiction.
%
% On constate que les méthodes de type Borda-Count sont toutes très proches
% les unes des autres, entre 96 et 98% de bonne classification en top 1.
% 
% La meilleure des méthodes de Borda-Count est sans conteste la méthode de
% Borda-Count avec poids, pondérée. Ce méthode associe a chaque rang un
% poids qui est $c^(r-1)$ où $c$ est une constante dans $[0,1]$ et $r$ le
% rang.
%
% La méthode du meilleur rang peut également être intéressante si on tolère
% un très fort taux de rejet en ambiguité. En effet, cette méthode produit
% très peu d'erreurs en top 1 (0,03% en test), mais rejette beaucoup (77%
% en test).

clear all
load data

lignes = {'BC moyenne app', 'BC moyenne test', ...
          'BC poids app', 'BC poids test', ...
          'BC moyenne pondéré app', 'BC moyenne pondéré test', ...
          'BC poids pondéré app', 'BC poids pondéré test', ...
          'Meilleur rang app', 'Meilleur rang test'};
colonnes = {'% Classification', '% Rejet', '% Confusion'};
results = cell(length(lignes), 2);

[results{1,1}, results{1,2}, ...
 results{3,1}, results{3,2}, ...
 results{5,1}, results{5,2}, ...
 results{7,1}, results{7,2}, ...
 results{9,1}, results{9,2}] = combinaisonRang(Xapp, Xapp, yapp{1});

[results{2,1}, results{2,2}, ...
 results{4,1}, results{4,2}, ...
 results{6,1}, results{6,2}, ...
 results{8,1}, results{8,2}, ...
 results{10,1}, results{10,2}] = combinaisonRang(Xtest, Xapp, ytest{1});

results = cell2mat(cellfun(@(x) x', results, 'UniformOutput', false));
results(:,11:15) = 1 - results(:,1:5) - results(:,6:10);

showTable(results(:,1:5:end)*100, colonnes, lignes);
showTable(results(:,5:5:end)*100, colonnes, lignes);

screen = get(0,'screensize');
f = figure('Position',[0,0,screen(3),screen(4)-100]); movegui(f,'northwest')
subplot(2,3,1);
plotResults(results(:,1:5)', 'classification', lignes)
subplot(2,3,2);
plotResults(results(:,6:10)', 'rejet', {})
title('Résultats pour toutes les méthodes');
subplot(2,3,3);
plotResults(results(:,11:15)', 'confusion', {})

subplot(2,3,4);
plotResults(results(1:8,1:5)', 'classification', {})
subplot(2,3,5);
plotResults(results(1:8,6:10)', 'rejet', {})
title('Résultats pour les méthodes Borda-count');
subplot(2,3,6);
plotResults(results(1:8,11:15)', 'confusion', {})

lignesRang = lignes;
resultsRang = results;
save('resultsRang', 'resultsRang', 'lignesRang');

%% Méthode de combinaison de type "Mesure"
%
% Essayons maintenant des méthodes de combinaison de type mesure. On
% utilise donc directement les scores en sortie des classifieurs, affectés
% à chaque classe pour chaque exemple.
% 
% Ces scores peuvent être combinés par somme ou produit, pondérés ou non.
%
% Dans notre cas, les méthodes de somme donnent des meilleurs résultats que
% le produit sur tous les plans : taux plus fort en classification et plus
% faible en rejet et en confusion.

clear all
load data

lignes = {'Somme app', 'Somme test', ...
          'Produit app', 'Produit test', ...
          'Somme pondéré app', 'Somme pondéré test', ...
          'Produit pondéré app', 'Produit pondéré test'};
colonnes = {'% Classification', '% Rejet', '% Confusion'};
results = cell(length(lignes), 2);

[results{1,1}, results{1,2}, ...
 results{3,1}, results{3,2}, ...
 results{5,1}, results{5,2}, ...
 results{7,1}, results{7,2}] = combinaisonMesure(Xapp, Xapp, yapp{1});

[results{2,1}, results{2,2}, ...
 results{4,1}, results{4,2}, ...
 results{6,1}, results{6,2}, ...
 results{8,1}, results{8,2}] = combinaisonMesure(Xtest, Xapp, ytest{1});

results = cell2mat(cellfun(@(x) x', results, 'UniformOutput', false));
results(:,11:15) = 1 - results(:,1:5) - results(:,6:10);

showTable(results(:,1:5:end)*100, colonnes, lignes);
showTable(results(:,5:5:end)*100, colonnes, lignes);

screen = get(0,'screensize');
f = figure('Position',[0,0,screen(3),screen(4)-400]); movegui(f,'northwest')
subplot(1,3,1);
plotResults(results(:,1:5)', 'classification', lignes)
subplot(1,3,2);
plotResults(results(:,6:10)', 'rejet', {})
subplot(1,3,3);
plotResults(results(:,11:15)', 'confusion', {})

lignesMes = lignes;
resultsMes = results;
save('resultsMes', 'resultsMes', 'lignesMes');

%% Comparaison des méthodes
% 
% On se propose finalement de comparer les performances des différentes
% méthodes en test (les résultats en apprentissage et en test étant
% quasiment identique, inutile de doubler la quantité de données à
% analyser).
%
% En top 1, on affiche un tableau des résultats, trié par taux de
% classification. On voit que la majorité des méthodes sont proches les
% unes des autres, mais que le vote pondéré donne les meilleurs résultats.
%
% Il peut être intéressant de chercher le front de Pareto des solutions à
% notre disposition, afin de savoir quelles sont réellement les solutions
% les plus interessantes au sens d'une optimisation multi-critère visant a
% maximiser le taux de classification et minimiser le taux de confusion
% (mathématique, cela revient également à minimiser le taux de rejet).
%
% On constate que le front de Pareto en top 1 contient le vote à la
% pluralité, le vote à la majorité, le vote pondéré et le meilleur rang.
% Ces résultats sont particulièrement étonnant puisqu'ils ne font
% apparaitre quasiment que des méthodes de type vote, et une méthode de
% type rang qui rejette énormément lui permettant d'avoir un taux
% imbattablement faible en confusion la plaçant dans le front.
%
% Cependant, on ne peut bien sûr pas généraliser ces résultats obtenus sur
% un cas particulier. Par ailleurs, il est important de noter que les
% différences entres les méthodes sont très faibles pour la majorité
% d'entre elles et que ce classement est donc peu significatif.
%
% Enfin, on peut également regarder l'évolution des performances des
% différentes méthodes du top 1 au top 5 (affiché uniquement pour les
% méthodes ayant un taux de classification supérieur à 95% afin que le
% graphe reste lisible). Globalement, les résultats restent très
% "parallèle", une méthode dépasse rarement une autre.

clear all;

% load results
load resultsMes
load resultsVote
load resultsRang
resultsVote = [repmat(resultsVote(:,1), 1, 5) repmat(resultsVote(:,2), 1, 5) repmat(resultsVote(:,3), 1, 5)];
colonnes = {'% Classification', '% Rejet', '% Confusion'};

% merge all
results = [resultsVote ; resultsRang ; resultsMes];
lignes = [lignesVote lignesRang lignesMes];

% remove app results
results = results(2:2:end, :);
lignes = lignes(2:2:end);

% show table
[~, inds] = sort(results(:,1), 'descend');
showTable(results(inds,1:5:end)*100, colonnes, lignes(inds));

% front pareto
[~, indsPareto] = prtp(results(:,6:5:end));
showTable(results(indsPareto,1:5:end)*100, colonnes, lignes(indsPareto));

% filter good enougth methods
inds = find(results(:,1) > .95);
results = results(inds, :);
lignes = lignes(inds);

% plot
colors = {'b', 'r', 'g', 'k'};
styles = {'-x', '--o'};
screen = get(0,'screensize');
f = figure('Position',[0,0,screen(3),screen(4)-400]); movegui(f,'northwest')
subplot(1,3,1);
plotResults(results(:,1:5)', 'classification', lignes, colors, styles)
subplot(1,3,2);
plotResults(results(:,6:10)', 'rejet', {}, colors, styles)
subplot(1,3,3);
plotResults(results(:,11:15)', 'confusion', {}, colors, styles)
