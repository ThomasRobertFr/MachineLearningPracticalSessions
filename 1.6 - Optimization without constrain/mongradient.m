function d = mongradient(theta)
    d = [-2*(1-theta(1)) - 400*theta(1)*(theta(2) - theta(1)^2) ; 
        200*(theta(2)-theta(1)^2)];
end