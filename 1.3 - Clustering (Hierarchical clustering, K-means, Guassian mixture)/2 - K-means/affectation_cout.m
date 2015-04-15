function [Jw, liste] = affectation_cout(M)
    [couts, liste] = min(M,[] , 2); % indices des minimums par ligne
    Jw = sum(couts);
end