function [lambdak, k] = lambdaMin(lambda)

lambda(lambda < 0) = Inf;
[lambdak, k] = min(lambda);