function M = distance(X, param)
% X : R ^ N x d
% N : Nb points
% M : R ^ N x N distance entre les points
    
    n = size(X,1);
    
    if (param == 'euclid')
        %M = zeros(n,n);
    
        %for i = 1:n
        %    for j = i+1:n
        %        dist = norm(X(i,:) - X(j,:));
        %        M(i,j) = dist;
        %        M(j,i) = dist;
        %    end
        %end

        %  Autre méthode :
        % 
        % ||x-y||² = ||x||² + ||y||² - 2 x'*y
        %
        
        produits = X*X'; % produits scalaires
        normes = diag(produits); % normes des vecteurs sur la diag
        M = normes*ones(1,n) + ones(n,1)*normes' - 2* produits;
    end
end