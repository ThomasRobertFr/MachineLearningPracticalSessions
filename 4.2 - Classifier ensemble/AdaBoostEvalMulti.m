function results = AdaBoostEvalMulti(Xapp, Yapp, Xtest, Ytest, B, verbose)

if (nargin < 6)
    verbose = false;
end

nbCl = max(Yapp);

% init
YappHats = zeros(size(Xapp,1), nbCl, B);
YtestHats = zeros(size(Xtest,1), nbCl, B);
YappI = Yapp;
YtestI = Ytest;

% 1 vs all
for i = 1:nbCl
    YappI(Yapp ~= i) = -1;
    YappI(Yapp == i) = 1;
    YtestI(Ytest ~= i) = -1;
    YtestI(Ytest == i) = 1;
    
    [YappHats(:, i, :), ~, YtestHats(:, i, :)] = AdaBoostEval(Xapp, YappI, Xtest, YtestI, B);
end

% combinaison et éval des perfs
YappHats(YappHats == -1) = 0;
YtestHats(YtestHats == -1) = 0;

classApp = zeros(1,B);
rejetApp = zeros(1,B);
classTest = zeros(1,B);
rejetTest = zeros(1,B);
for i = 1:B
    [classApp(i), rejetApp(i)] = evaluerPerfs(squeeze(YappHats(:,:,i)), Yapp, 1);
    [classTest(i), rejetTest(i)] = evaluerPerfs(squeeze(YtestHats(:,:,i)), Ytest, 1);
end
errorTest = 1 - classTest - rejetTest;

results = [classTest' rejetTest' errorTest']*100;

if (verbose)
    figure;
    plot(results);
    legend('Classification', 'Rejet', 'Confusion', 'Location', 'best');
    xlabel('Nombre de classifieurs L');
    ylabel('Taux (%)');
end