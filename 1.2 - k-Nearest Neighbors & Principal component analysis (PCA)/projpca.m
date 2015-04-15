function Ct = projpca(Z, moy, P)
    [n, p] = size(Z);

    if (mean(mean(Z)) < 1e-8)
        Z = Z - ones(n, 1)*moy;
    end
    
    Ct = Z*P;
end