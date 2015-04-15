%% 
close all;
clc;
clear all;

load gauss3.mat;
%X = data;
X = load('george.dat');
X = mydownsampling(X, 7);

[N, d] = size(X);
K = 6;

figure();
hold on;

% Initialisation
mu = init_centres(X, K);
S = {};
for j = 1:K
    S{j} =  eye(d)*var(X(:,1));

end
pi = ones(K,1)*1/K;

mu_prec = mu + 1;
while abs(sum(mu-mu_prec)) > 1e-6
    mu_prec = mu;
    
    % Etape E

    P = zeros(N,K);
    for j = 1:K
        P(:,j) = pi(j) * mvnpdf(X, mu(j,:), S{j});
    end
    P = P./repmat(sum(P,2), 1, K);

    % Etape M

    mu = (X'*P./repmat(sum(P),d,1))';
    %mu = zeros(K, d);
    %for j = 1:K
    %    mu(j,:) = sum(repmat(P(:,j), 1, d).*X)/sum(P(:,j));
    %end

    pi = sum(P)/N;
    %pi = zeros(K, 1);
    %for j = 1:K
    %    pi(j) = sum(P(:,j))/N;
    %end

    for j = 1:K
        S{j} =  (X - repmat(mu(j,:),N,1))'*diag(P(:,j))*(X - repmat(mu(j,:),N,1)) / sum(P(:,j));
        
        %S{j} = zeros(d);
        %for i = 1:N
        %   S{j} = S{j} + P(i,j)*(X(i,:)-mu(j,:))'*(X(i,:)-mu(j,:));
        %end
        %S{j} = S{j} / sum(P(:,j));
    end

    plot(mu(:,1), mu(:,2), '+','Color',[0.5,0.5,0.5]);

end

% extraction de la liste des clusters
[~, liste] = max(P');
% affichage des clusters
show_clusters(X, liste);
% affichage des centres
scatter(mu(:,1), mu(:,2), 50, 'd', 'k', 'fill');

