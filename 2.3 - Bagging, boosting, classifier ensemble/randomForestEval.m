function [tauxErreurTrain, tauxErreurTest, stdTrain, stdTest] = randomForestEval(X, Y, Xtest, Ytest, hmax, purete, nbArbres, nbDim, nbIte)

if (nargin < 9)
    nbIte = 20;
end

tauxErreurTrainMat = zeros(1,nbIte);
tauxErreurTestMat = zeros(1,nbIte);

for i=1:nbIte
    forest = randomForestTrain(X, Y, hmax, purete, nbArbres, nbDim);
    
    Yhat = randomForestVal(forest, X);
    tauxErreurTrainMat(i) = sum(Yhat ~= Y) / length(Y);

    Ytesthat = randomForestVal(forest, Xtest);
    tauxErreurTestMat(i) = sum(Ytesthat ~= Ytest) / length(Ytest);
end

tauxErreurTrain = mean(tauxErreurTrainMat);
tauxErreurTest = mean(tauxErreurTestMat);
stdTrain = std(tauxErreurTrainMat);
stdTest = std(tauxErreurTestMat);