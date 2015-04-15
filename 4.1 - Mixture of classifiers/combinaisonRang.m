function [txClassBCMean, txRejetBCMean, ...
          txClassBCWeighted, txRejetBCWeighted, ...
          txClassBCMeanPond,     txRejetBCMeanPond, ...
          txClassBCWeightedPond, txRejetBCWeightedPond, ...
          txClassBestRank, txRejetBCBestRank] = ...
             combinaisonRang(Xs, Xapp, y)

n = size(Xs{1}, 1);
p = size(Xs{1}, 2);
nbCl = length(Xs);

% init
bordaCountMean = zeros(n, p);
bordaCountWeighted = zeros(n, p);
bordaCountMeanPond = zeros(n, p);
bordaCountWeightedPond = zeros(n, p);
bestRank = zeros(n, p);
c = 0.67;

% pour chaque classifieur
for k = 1:nbCl
    X = Xs{k};
    poids = evaluerPerfs(Xapp{k}, y, 1);
    bestRankForCl = zeros(n, p);
    
    % pour chaque position de vote
    for i = 1:5
        % selection de la classe votée en position i pour chaque obs
        indsI = (1:n)';
        [probas, indsJ] = max(X, [], 2);
        
        % on retire les classes non votées (si moins de i votes)
        indsI = indsI(probas > 0);
        indsJ = indsJ(probas > 0);
        
        % on clacule les indices à mettre à jour
        inds = sub2ind(size(X), indsI, indsJ);
        
        % on incrémente les borda count selon les votes
        bordaCountMean(inds)         = bordaCountMean(inds)         + (6 - i);
        bordaCountMeanPond(inds)     = bordaCountMeanPond(inds)     + poids*(6 - i);
        bordaCountWeighted(inds)     = bordaCountWeighted(inds)     + c^(i-1);
        bordaCountWeightedPond(inds) = bordaCountWeightedPond(inds) + poids*c^(i-1);
        
        % on enregistre les rangs
        bestRankForCl(inds) = 6 - i;
        
        % on supprime ce vote qui a été traité
        X(inds) = 0;
    end
    
    bestRank = max(bestRank, bestRankForCl);
end

% resultats
[txClassBCMean,         txRejetBCMean        ] = evaluerPerfs(bordaCountMean, y);
[txClassBCWeighted,     txRejetBCWeighted    ] = evaluerPerfs(bordaCountWeighted, y);
[txClassBCMeanPond,     txRejetBCMeanPond    ] = evaluerPerfs(bordaCountMeanPond, y);
[txClassBCWeightedPond, txRejetBCWeightedPond] = evaluerPerfs(bordaCountWeightedPond, y);
[txClassBestRank,       txRejetBCBestRank    ] = evaluerPerfs(bestRank, y);

