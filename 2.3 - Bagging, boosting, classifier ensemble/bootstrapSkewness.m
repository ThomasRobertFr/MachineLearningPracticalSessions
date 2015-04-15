% Chargement CSV
X = csvread('skewnormal.csv');
n = length(X);
B = 50;

% On calcule B skewness
skewVals = zeros(B,1);
for i = 1:B
    skewVals(i) = skewness(X(tireBootstrap(n,n)));
end
skew = mean(skewVals)
errSkew = std(skewVals)
