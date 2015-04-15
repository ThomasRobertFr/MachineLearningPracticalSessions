function [w, b, alpha] = monsvmclass(X, labels, C)


    n = size(X,1);
    K = X*X';
    Y = diag(labels);
    H = Y*K*Y;
    q = ones(n, 1);
    alpha = zeros(n,1);

    [alphaNotNull, lambda, pos, mu] = monqp(H, q, labels, 0, C*q, 1e-6, 1);
    
    alpha(pos) = alphaNotNull;
    w = alpha'*Y*X;
    b = lambda;
    
end