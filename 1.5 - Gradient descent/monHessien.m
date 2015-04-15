function H = monHessien(a, b, c, theta)
    H = exp(-0.1)*(exp(a'*theta)*(a*a') + exp(b'*theta)*(b*b') + exp(c'*theta)*(c*c'));
end