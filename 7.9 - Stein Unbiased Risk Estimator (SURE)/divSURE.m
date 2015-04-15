function div = divSURE(S, a, lambda, f, fderive)

[n, p] = size(S);
s = diag(S);

div = sum(fderive(s, a, lambda) + (n - p) * f(s, a, lambda) ./ s);

doubleSum = 1 ./ (repmat(s.^2, 1, p) - repmat(s.^2', p, 1)); % compute 1 / si^2 - sj^2
doubleSum(logical(eye(size(doubleSum)))) = 0; % remove elemnts on diag
doubleSum = doubleSum .* repmat(s .* f(s, a, lambda), 1, p); % compute si * f(si) / si^2 - sj^2 
doubleSum = sum(sum(doubleSum));

div = div + 2 * doubleSum;