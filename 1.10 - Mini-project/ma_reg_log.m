function [theta,L,p] = ma_reg_log(phi, z)
    
    theta = zeros(size(phi,2), 1);
    theta_old = theta + 5;
    i=1;
    
    L = [];
    
    while norm(theta_old - theta) > 1e-1
        theta_old = theta;
        p = probaAPosteriori(theta, phi);
        l = -sum(z.*log(p) + (1 - z).*log(1-p));
        L = [L l];
        W = diag(p.*(1-p));
        r =  phi*theta + W\(z - p);
        theta = (phi'*W*phi)\(phi'*W*r);
        i = i + 1;
    end
end

