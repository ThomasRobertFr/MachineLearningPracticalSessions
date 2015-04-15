function Jw = cout(M, liste)

    [N, K] = size(M);

    Jw = 0;
    
    % pour chaque cluster 
    for k = 1:K
        % distance intra-cluster du cluster k
        Jw = Jw + sum(M(liste == k, k));
    end
    
    % Jw = 0;
    % for i = 1:N
    %     Jw = Jw + M(i, liste(i));
    % end
end