function [txClassPlur, txRejetPlur, ...
          txClassMaj,  txRejetMaj, ...
          txClassPond, txRejetPond] = ...
            combinaisonClasse(Xs, Xapp, y)

n = length(y);
p = max(y);
nbCl = length(Xs);

yHats = zeros(n,nbCl);

for i = 1:nbCl
    [~, yHats(:,i)]  = max(Xs{i}, [], 2);
end

[yHat, yHatFreq, yHatConccurents] = mode(yHats, 2);
yHatNbConccurents = cellfun(@length, yHatConccurents);

% vote a la pluralité
yHatPlur = yHat;
yHatPlur(yHatNbConccurents > 1) = 0;

% vote a la majorité
yHatMaj = yHatPlur;
yHatMaj(yHatFreq < nbCl/2) = 0;

% vote pondéré
yHatsPond = zeros(n, p);
for i = 1:nbCl
    poids = evaluerPerfs(Xapp{i}, y, 1);
    inds = sub2ind(size(yHatsPond), (1:n)', yHats(:,i));
    yHatsPond(inds) = yHatsPond(inds) + poids;
end
[yHatPondVotes, yHatPond] = max(yHatsPond, [], 2);
yHatPondNbConcurrents = sum(repmat(yHatPondVotes, 1, p) == yHatsPond, 2);
yHatPond(yHatPondNbConcurrents > 1) = 0;

% taux

txClassPlur = sum(yHatPlur == y)/n;
txRejetPlur = sum(yHatPlur == 0)/n;

txClassMaj = sum(yHatMaj == y)/n;
txRejetMaj = sum(yHatMaj == 0)/n;

txClassPond = sum(yHatPond == y)/n;
txRejetPond = sum(yHatPond == 0)/n;
























