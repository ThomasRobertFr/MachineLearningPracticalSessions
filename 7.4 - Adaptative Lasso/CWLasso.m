function b = CWLasso(X, y, lambda, b)

p = size(X, 2);

% init iteration variables
JOld = 0;
J = Inf;
i = 0;

% while we don't iterate too much (avoid infinite loop) & the cost varies
while (i < 10000 && abs(JOld - J) > 10e-7)
    
    % go through all p's
    for pj = randperm(p)
        
        % Compute bj_MC for j-th component
        x = X(:,pj);
        zInds = setdiff(1:p,pj);
        z = y - X(:,zInds)*b(zInds);

        bjMC = (x'*z)/(x'*x);

        % compute bj
        b(pj) = sign(bjMC)*max(0, abs(bjMC)-lambda/(x'*x));
    end
    
    % compute J (cost)
    JOld = J;
    J = norm(X*b - y)^2;
    
    % increment i
    i = i + 1;
end
