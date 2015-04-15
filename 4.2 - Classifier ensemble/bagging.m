%% Bagging pima

disp('Bagging - Pima (souche)')

clear all
load pima
y(y==-1) = 2;

tauxErreurOOBMoy = [];
ecartErreurOOB = [];
tauxErreurTestBagging = [];
tauxErreurTestSansBagging = []; 

for i = 1:20
    [Xapp, Yapp, Xtest, Ytest] = splitdata(x, y, 0.1);
    [tauxErreurOOBMoy(end+1), ecartErreurOOB(end+1), tauxErreurTestBagging(end+1), tauxErreurTestSansBagging(end+1)] = ...
        baggingEval(Xapp, Yapp, Xtest, Ytest, 10, @threshold_tr, @threshold_te);
end

tauxErreurOOBMoy = mean(tauxErreurOOBMoy)
ecartErreurOOB = mean(ecartErreurOOB)
tauxErreurTestBagging = mean(tauxErreurTestBagging)
tauxErreurTestSansBagging = mean(tauxErreurTestSansBagging)

%% Bagging satimage

disp('Bagging - Satimage (1 ppv)')
                        
clear all
load satimage

tauxErreurOOBMoy = [];
ecartErreurOOB = [];
tauxErreurTestBagging = [];
tauxErreurTestSansBagging = [];

for i = 1:10
    [xapp, yapp, xtest, ytest] = splitdata(x, y, 0.1);
    [tauxErreurOOBMoy(end+1), ecartErreurOOB(end+1), tauxErreurTestBagging(end+1), tauxErreurTestSansBagging(end+1)] = ...
        baggingEval(xapp, yapp, xtest, ytest, 30, @oneppv_tr, @oneppv_te);
end

tauxErreurOOBMoy = mean(tauxErreurOOBMoy)
ecartErreurOOB = mean(ecartErreurOOB)
tauxErreurTestBagging = mean(tauxErreurTestBagging)
tauxErreurTestSansBagging = mean(tauxErreurTestSansBagging)

%% pima boosting

clear all
load pima

%[Xapp, Yapp, Xtest, Ytest] = splitdata(x, y, 0.7);
%AdaBoostEval(Xapp, Yapp, Xtest, Ytest, 10, true);
%title('Performances pima par AdaBoost');

tauxErreurTest = [];

for i = 1:10
    [Xapp, Yapp, Xtest, Ytest] = splitdata(x, y, 0.7);
    [~, ~, ~, tauxErreurTest(i,:)] = AdaBoostEval(Xapp, Yapp, Xtest, Ytest, 10);
end
tauxErreurTest = mean(tauxErreurTest);
plot(tauxErreurTest);
title('Performances moyennes pima par AdaBoost');
xlabel('Nombre de classifieurs');
ylabel('Taux d''erreur (%)');

%% satimage boosting

clear all
load satimage
y(y == 7) = 6;

results = [];

for i = 1:10
    [Xapp, Yapp, Xtest, Ytest] = splitdata(x, y, 0.5);
    results(:,:,i) = AdaBoostEvalMulti(Xapp, Yapp, Xtest, Ytest, 30);
end

results = mean(results, 3);
plot(results);
title('Performances moyennes satimage par AdaBoost');
legend('Classification', 'Rejet', 'Confusion', 'Location', 'best');
xlabel('Nombre de classifieurs');
ylabel('Taux (%)');

%% RSM pima

disp('RSM - Pima (souche)')

clear all
load pima
y(y==-1) = 2;

tauxErreurAppMoy = [];
ecartErreurApp = [];
tauxErreurTestRSM = [];
tauxErreurTestSansRSM = [];

for i = 1:20
    [Xapp, Yapp, Xtest, Ytest] = splitdata(x, y, 0.5);
    [tauxErreurAppMoy(end+1), ecartErreurApp(end+1), tauxErreurTestRSM(end+1), tauxErreurTestSansRSM(end+1)] = ...
        randomSubspaceEval(Xapp, Yapp, Xtest, Ytest, 10, 3, @threshold_tr, @threshold_te);
end

tauxErreurAppMoy = mean(tauxErreurAppMoy)
ecartErreurApp = mean(ecartErreurApp)
tauxErreurTestRSM = mean(tauxErreurTestRSM)
tauxErreurTestSansRSM = mean(tauxErreurTestSansRSM)

%% RSM satimage

disp('RSM - Satimage (1 ppv)')

clear all
load satimage

tauxErreurAppMoy = [];
ecartErreurApp = [];
tauxErreurTestRSM = [];
tauxErreurTestSansRSM = [];

for i = 1:10
    [xapp, yapp, xtest, ytest] = splitdata(x, y, 0.3);
    [tauxErreurAppMoy(end+1), ecartErreurApp(end+1), tauxErreurTestRSM(end+1), tauxErreurTestSansRSM(end+1)] = ...
        randomSubspaceEval(xapp, yapp, xtest, ytest, 30, 20, @oneppv_tr, @oneppv_te);
end

tauxErreurAppMoy = mean(tauxErreurAppMoy)
ecartErreurApp = mean(ecartErreurApp)
tauxErreurTestRSM = mean(tauxErreurTestRSM)
tauxErreurTestSansRSM = mean(tauxErreurTestSansRSM)

