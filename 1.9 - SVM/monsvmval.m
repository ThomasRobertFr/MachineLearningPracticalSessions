function y = monsvmval(X, w, b)
    y = w'*X'+b;
    
    y = y';
    
    y(y >= 0) = 1;
    y(y < 0) = -1;
end