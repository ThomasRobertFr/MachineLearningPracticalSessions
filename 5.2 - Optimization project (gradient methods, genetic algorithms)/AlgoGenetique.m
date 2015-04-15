function ti = AlgoGenetique(param, pi)

nbPts = 200;

tis = linspace(0, 1, nbPts);
mtis = (param'*[tis.^3; tis.^2; tis; ones(1, length(tis))])';
Js = sum((repmat(pi', length(tis), 1) - mtis).^2, 2);

alpha = 0.2;

J = min(Js);
Jold = Inf;
nbGoodIte = 0;

while (abs(J - Jold) > 1e-6 && nbGoodIte < 4)
    
    if (abs(J - Jold) > 1e-6)
        nbGoodIte = nbGoodIte + 1;
    else
        nbGoodIte = 0;
    end
    
    % mutation
    mtis = (param'*[tis.^3; tis.^2; tis; ones(1, length(tis))])';
    Js = sum((repmat(pi', length(tis), 1) - mtis).^2, 2);
    [~, is] = sort(Js);
    for i = 1:2:nbPts
        gamma = (1 + 2 * alpha) * rand - alpha;
        tis(end + 1) = (1 - gamma) * tis(is(i)) + gamma * tis(is(i+1));
    end

    % réduction
    mtis = (param'*[tis.^3; tis.^2; tis; ones(1, length(tis))])';
    Js = sum((repmat(pi', length(tis), 1) - mtis).^2, 2);
    [~, is] = sort(Js);
    tis = tis(is(1:nbPts));

    % cout
    Jold = J;
    J = min(Js);
end

mtis = (param'*[tis.^3; tis.^2; tis; ones(1, length(tis))])';
Js = sum((repmat(pi', length(tis), 1) - mtis).^2, 2);
[~, ind] = min(Js);
ti = tis(ind);

