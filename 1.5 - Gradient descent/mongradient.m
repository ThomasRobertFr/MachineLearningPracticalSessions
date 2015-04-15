function d = mongradient(a, b, c, theta)
    d = exp(-0.1)*(a*exp(a'*theta) + b*exp(b'*theta) + c*exp(c'*theta));
end