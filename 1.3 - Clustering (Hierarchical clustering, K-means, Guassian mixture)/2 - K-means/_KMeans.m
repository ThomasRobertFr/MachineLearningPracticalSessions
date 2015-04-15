% DM / TP4 / MUSSARD / ROBERT

clear all
close all

% Liste des méthodes a appliquer.
% Pour chaque méthode, liste des fichers de données ainsi que des
% paramètres de test.
methodes = struct(                                                  ...
    'methode',{'euclid'; 'mahal'},                                  ...
    'tests', {                                                      ...
        struct(                                                     ...
           'file', {'ds2.dat' ; 'ds3.dat' ; 'ds4.dat' ; 'ds5.dat' ; 'george.dat'}, ...
           'Klist',{2:7 ; 2:6 ; 4:9 ; 3:8 ; 2:6},                   ...
           'nbIte',{3 ; 3 ; 3 ; 3 ; 3});                            ...
        struct(                                                     ...
           'file', {'ds2.dat' ; 'ds3.dat' ; 'ds4.dat' ; 'ds5.dat' ; 'george.dat'}, ...
           'Klist',{2:7 ; 2:6 ; 4:9 ; 3:8 ; 2:6},                   ...
           'nbIte',{3 ; 3 ; 3 ; 3 ; 3})                             ...
        });

% pour chaque méthode (euclid, mahal, ...)
for i = 1:length(methodes)
    
    methode = methodes(i).methode;
    tests = methodes(i).tests;
    
    % pour chaque jeu de données
    for j = 1:length(tests)
        
        % init des données
        X = load(tests(j).file);
        Klist = tests(j).Klist;
        nbIte = tests(j).nbIte;
        
        % init des figures
        figClusters = figure();
        figJw = figure();
        
        % execution du test
        test_k_means(X, Klist, nbIte, methode, figClusters, figJw);
        
        % sauvegarde des figures
        set(figClusters, 'PaperUnits', 'points');
        set(figClusters, 'PaperPosition', [0 0 1000 1000]);
        saveas(figClusters, ['TP4_MUSSARD_ROBERT_' int2str(figClusters) '.png']);
        set(figJw, 'PaperUnits', 'points');
        set(figJw, 'PaperPosition', [0 0 500 600]);
        saveas(figJw, ['TP4_MUSSARD_ROBERT_' int2str(figJw) '.png']);
        
    end
end