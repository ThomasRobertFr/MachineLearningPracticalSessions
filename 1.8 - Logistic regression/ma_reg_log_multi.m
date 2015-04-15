function [theta,L,p] = ma_reg_log_multi(phi, z)
    
    nbFront = size(z, 2);
    
    theta = zeros(size(phi,2), nbFront)+1;
    theta_old = theta + 5;
    i=1;
    
    L = [];
    
    %while norm(theta_old - theta) > 1e-1
    for i = 1:100
        theta_old = theta;
        p = probaAPosteriori(theta, phi);
        %l = -sum(z.*log(p) + (1 - z).*log(1-p));
        %L = [L l];
        for j = 1:nbFront
            W = diag(p(:,j) .* (1 - p(:,j)));
            r =  phi*theta(:,j) + W\(z(:,j) - p(:,j));
            theta(:,j) = (phi'*W*phi)\(phi'*W*r);
        end
        %i = i + 1;
    end
end

