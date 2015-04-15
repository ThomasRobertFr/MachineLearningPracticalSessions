function [z, nbClasse] = EncoderLabel01_Multi(y)

    vals = unique(y);
    nbClasse = length(vals);
    
    z = zeros(size(y, 1), nbClasse - 1);

    for i = 1:nbClasse - 1
        z(y == vals(i),i) = 1;
    end
    
end