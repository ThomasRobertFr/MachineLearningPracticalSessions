function p = probaAPosteriori(theta, phi)
    
    p = exp(phi*theta);
    p = p./repmat((1+sum(p,2)), 1, size(theta,2));
    
end