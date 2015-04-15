function [tops, rejets] = evaluerPerfs(X, y, maxTop, order)

% init
if (nargin < 3)
    maxTop = 5;
end
if (nargin < 4)
    order = 'descend';
end
%%

n = size(X,1);
p = size(X,2);
tops = zeros(maxTop,1);
rejets = zeros(maxTop,1);

% sort X
[yProbas, yClasses] = sort(X, 2, order);

% nb predicted = nb de col sans zéro
nbPredicted = sum(yProbas ~= 0, 2);

% rand de prédiction des bons résultats
[~, yHatInd] = sort((repmat(y, 1, p) == yClasses), 2, 'descend');
yHatInd = yHatInd(:,1);

for i = 1:maxTop
    % positions en rejets possible = conflits avec la proba suivante
    rejetsPos = yProbas(:, i) == yProbas(:, i+1) & yProbas(:, i) > 0;
    
    % zone en conflit (zone de candidats top i)
    [~, zoneConflitDebut] = max(yProbas == repmat(yProbas(:, i), 1, p), [], 2);
    zoneConflitDebut(~rejetsPos) = Inf;
    [~, zoneConflitFin] = max(fliplr(yProbas == repmat(yProbas(:, i), 1, p)), [], 2);
    zoneConflitFin = p - zoneConflitFin + 1;
    zoneConflitFin(~rejetsPos) = Inf;
    
    % indicateurs de décision
    yInTopI = yHatInd <= i;
    yProbaSupZero = yHatInd <= nbPredicted;
    yNotInConflict = yHatInd < zoneConflitDebut;
    yInConflictZone = yHatInd >= zoneConflitDebut & yHatInd <= zoneConflitFin;
    
    % bonne classification finale
    yOk = yInTopI & yProbaSupZero & yNotInConflict;
    yRejet = yProbaSupZero & yInConflictZone | (nbPredicted == 0);
    
    % top & rejet
    tops(i) = sum(yOk) / n;
    rejets(i) = sum(yRejet) / n;
    
end
