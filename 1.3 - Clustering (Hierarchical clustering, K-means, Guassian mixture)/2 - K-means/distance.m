function M = distance(X, C, param)
    
    N = size(X,1);
    K = size(C,1);
    
    % M = zeros(N,K);
    % 
    % for i = 1:N
    %   for k = 1:K
    %       if (param == 'euclid')
    %           M(i,k) = norm(X(i,:) - C(k,:));
    %       end
    %   end
    % end
    
    if (strcmp(param, 'euclid'))
        normX = sum(X.^2, 2);
        normC = sum(C.^2, 2);
        ps = X*C';
        M = repmat(normX, 1, K) + repmat(normC', N, 1) - 2*ps;
    end
    
    if (strcmp(param, 'mahal'))
        
        M = zeros(N,K);
        
        for k = 1:K
            X2 = X + repmat(mean(X),N,1) - repmat(C(k,:),N,1); 
            M(:, k) = mahal(X2, X);
        end
    end
end