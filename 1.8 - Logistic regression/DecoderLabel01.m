function y = DecoderLabel01(z)
    
    z = [z 1 - sum(z,2)];
    [~, y] = max(z, [], 2);
    
end
    
    
    
    