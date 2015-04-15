function cout = moncritere(a, b, c, theta)
    cout = exp(-0.1)*(exp(a'*theta) + exp(b'*theta) + exp(c'*theta));
end