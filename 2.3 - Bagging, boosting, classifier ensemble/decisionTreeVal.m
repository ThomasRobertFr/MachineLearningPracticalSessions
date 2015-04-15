function Yhat = decisionTreeVal(tree, X)

Yhat = souchebinaireval(tree.decision, X);

% si fils gauche existant, recalcul
if (isfield(tree, 'filsGauche'))
    Yhat(Yhat == -1) = decisionTreeVal(tree.filsGauche, X(Yhat == -1, :));
end

% si fils droit existant, recalcul
if (isfield(tree, 'filsDroit'))
    Yhat(Yhat == 1) = decisionTreeVal(tree.filsDroit, X(Yhat == 1, :));
end