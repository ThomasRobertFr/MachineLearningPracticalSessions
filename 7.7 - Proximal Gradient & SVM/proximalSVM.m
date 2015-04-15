function [w, Js] = proximalSVM(X, y, rho, lambda)

p = size(X,2);

i = 1;
w = zeros(p,1);
Y = diag(y);

JOld = Inf;
J = 1/2 * sum(max(0,1-Y*X*w).^2)+lambda * norm(w,1);

if (nargout > 1)
    Js = J;
end

while (JOld - J > 1e-7 && i < 1e3)
    
    % try compute w
    gradL = -(Y*X)'*max(0, 1 - Y*X*w);
    whatNew = w - rho * gradL;
    wNew = sign(whatNew) .* max(0, abs(whatNew) - rho * lambda);
    wNew(end) = whatNew(end); % restore w0
    
    % cost
    JNew = 1/2 * sum(max(0,1-Y*X*wNew).^2)+lambda * norm(wNew,1);
    
    % save if cost decrease
    if (JNew < J)
        w = wNew;
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
