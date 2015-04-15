function [txClassSum,      txRejetSum, ...
          txClassProd,     txRejetProd, ...
          txClassSumPond,  txRejetSumPond, ...
          txClassProdPond, txRejetProdPond] = combinaisonMesure(Xs, Xapp, y)

n = size(Xs{1}, 1);
p = size(Xs{1}, 2);
nbCl = length(Xs);

Xsum = zeros(n, p);
Xprod = ones(n, p);
XsumPond = Xsum;
XprodPond = Xprod;

for i = 1:nbCl
    poids = evaluerPerfs(Xapp{i}, y, 1);
    
    Xsum = Xsum + Xs{i};
    Xprod = Xprod .* Xs{i};
    XsumPond = XsumPond + poids * Xs{i};
    XprodPond = XprodPond .* (Xs{i}.^poids);    
end

[txClassSum,      txRejetSum     ] = evaluerPerfs(Xsum, y);
[txClassProd,     txRejetProd    ] = evaluerPerfs(Xprod, y);
[txClassSumPond,  txRejetSumPond ] = evaluerPerfs(XsumPond, y);
[txClassProdPond, txRejetProdPond] = evaluerPerfs(XprodPond, y);
