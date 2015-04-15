clear
close all

load netflix_data_probe.mat
load netflix_data_app.mat

%% 
% L'objectif du TP est d'étudier le problème Netflix en essayant de
% déterminer la note qu'une personne attribuerait à une film a partir des
% notes qu'elle a attribuée aux autres films.
%
% Pour cela, on applique une méthode factorielle qui consiste donc à
% factoriser la matrice de données sous la forme d'un produit de 2 matrices
% plus petites $U$ et $V$.
% 
% Pour cela, on calcule les $k$ premiers vecteurs singuliers de la matrice.
% On utilisera pour cela la fonciton |lansvd| de PROPACK plutot que |svds|
% de Matlab pour des raisons d'optimisation.
%
% On estime ensuite les notes à "deviner" grâce à $U$ et $V$ et on les
% compare aux données de tests pour évaluer la qualité de la méthode.
%
% On constate que plus on prends de vecteurs singuliers, plus les résultats
% sont bon, jusqu'à 30. Je n'ai pas testé plus loin pour des raisons de
% temps de calculs, mais il semblerai selon les résultats du challenge
% Netflix qu'il faille prendre beaucoup de vecteurs propres pour commencer
% à faire du sur-apprentissage.
%
% En plus de quelques amélioration mémoire dans le calcul de l'erreur, j'ai
% essayé d'implémenter une méthode de soft-shrinkage. Malheureusement,
% cette méthode n'a apporté aucune modification de l'erreur supérieure à
% $10^{-14}$, donc rien de significant.

% nombre de vecteurs singuliers
k = 50;

% Calcul des vecteurs singuliers
tic
[U,D,V] = svds(netflix_data_app, k);
disp(['Time to compute SVD with svds : ' num2str(toc) 's']);
U=0;V=0;D=0; % clear RAM
tic
[U, D, V] = lansvd(netflix_data_app, k, 'L');
diagD = diag(D);
disp(['Time to compute SVD with propack : ' num2str(toc) 's']);

% recherche des éléments non nuls dans probe (éléments à estimer)
[i,j,s] = find(netflix_data_probe);
nt = length(s);

% Reconstructions
for nbVS = 1:k;
    
    % hard shrinkage
    tic
    Err(nbVS) = 0;
    d = D(1:nbVS,1:nbVS);
    for ii=1:nt
        rec = U(i(ii),1:nbVS)*d*V(j(ii),1:nbVS)';
        err = (rec - s(ii))^2;
        Err(nbVS) = Err(nbVS) + err;
    end
    Err(nbVS) = Err(nbVS) / nt;
    disp(['Time to reconstruct for ' num2str(nbVS) ' rank without soft-shrinkage : ' num2str(toc) 's - Err : ' num2str(Err(nbVS))]);
    
    % soft shrinkage
    if (nbVS < k)
        ErrSS(nbVS) = 0;
        tic
        d = diagD(1:nbVS);
        d = diag(soft_shrinckage(d, D(nbVS+1), d(ceil(nbVS*2/3))));
        for ii=1:nt
            rec = U(i(ii),1:nbVS)*d*V(j(ii),1:nbVS)';
            err = (rec - s(ii))^2;
            ErrSS(nbVS) = ErrSS(nbVS) + err;
        end
        ErrSS(nbVS) = ErrSS(nbVS) / nt;
        disp(['Time to reconstruct for ' num2str(nbVS) ' rank with    soft-shrinkage : ' num2str(toc) 's - Err : ' num2str(ErrSS(nbVS))]);
        
    end
    
end

% Evolution de l'erreur
figure;
plot(Err);
title('Evolution de l''erreur en fonction du rang');

