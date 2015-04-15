function X = soft_shrinckage(X, shrinkVal, maxVal)

n = length(X);

X(X < shrinkVal) = 0;

X(X >= shrinkVal & X <= maxVal) = ...
    (X(X >= shrinkVal & X <= maxVal) - shrinkVal) * maxVal / (maxVal - shrinkVal);
