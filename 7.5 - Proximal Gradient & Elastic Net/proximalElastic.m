function [b, Js] = proximalElastic(X, y, rho, lambda, mu)

p = size(X,2);

b = zeros(p,1);

XX = X'*X;
Xy = X'*y;

i = 1;
JOld = Inf;
J = 1/2*norm(X*b-y,2)^2 + lambda * norm(b, 1) + mu / 2 * norm(b)^2;

if (nargout > 1)
    Js = J;
end

while (JOld - J > 1e-7 && i < 1e6)
    
    % try compute beta
    bhatNew = b - rho * (XX*b - Xy + mu*b);
    bNew = sign(bhatNew) .* max(0, abs(bhatNew) - rho * lambda);

    % cost
    JNew = 1/2*norm(X*bNew-y,2)^2 + lambda * norm(bNew, 1) + mu / 2 * norm(bNew)^2;
    
    % save if cost decrease
    if (JNew < J)
        b = bNew;
        bhat = bhatNew;
        JOld = J;
        J = JNew;
        rho = rho * 1.15;
        
        if (nargout > 1)
            Js = [Js J];
        end
    % forget if cost increase
    else
        rho = rho / 2;
    end
    
    i = i + 1;
end

