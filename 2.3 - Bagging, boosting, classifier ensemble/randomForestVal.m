function Yhat = randomForestVal(forest, X)

[n,p] = size(X);
L = length(forest);

Yhats = zeros(n,L);

for i = 1:L
    Yhats(:,i) = decisionTreeVal(forest{i}.tree, X(:,forest{i}.features));
end

Yhat = mode(Yhats,2);