clear all;
close all;

load YaleFace/allYaleFaces

[n,p] = size(X);

%% SVD
%
% First, let's try to factorize $X$ matrix using SVD method.
%
% We get a $K$ rank factorization, however, how to interpret negative
% values?

% compute 16 first SVD
K = 16;
[U, S, V] = svds(X, K);

% estimate
Xhat = U*S*V';

% plot
i = 1;
for k = randperm(n, 8)
    if (i <= 4) subplot(4,4,i);
    else        subplot(4,4,4 + i); end
    imshow(reshape(X(k,:), 50, 50), [])
    title(['Original n°' int2str(k)])
    if (i <= 4) subplot(4,4,4 + i);
    else        subplot(4,4,8 + i); end
    imshow(reshape(Xhat(k,:), 50, 50), [])
    title(['Estimé n°' int2str(k)])
    i = i + 1;
end

figure
for i = 1:K
    subplot(4,ceil(K/4),i)
    imshow(reshape(V(:,i), 50, 50), [])
    title(['Composante V_{' int2str(i) '}'])
end


%% NNMF
%
% To solve this problem of negative values, let's use Non-negative Matrix
% Factorization of the same data.
%
% We will use proximal gradient method or this.
%
% Because of this constrain of positiveness, we can no longer find
% components with values that cancels each other. Therefore, the new
% components contains more null values on some zones of the picture, so
% that components add themselves and represent differents parts of the
% face.

% Params
K = 16;
eps = 1e-1;
kmax = 10000;
rhoU = 1e-5;
rhoV = 1e-5;

% Init
U = zeros(n,K);
V = rand(p,K);
J = Inf;
Js = [];

% Iterate
k = 1;
while (J > eps && k < kmax)
    VOld = V;
    
    for i = 1:n
        U(i,:) = max(0, U(i,:)' - rhoU * V'*(V*U(i,:)' - X(i,:)'));
    end

    for j = 1:p
        V(j,:) = ( V(j,:)' - rhoV * (U'*(U*V(j,:)' - X(:,j))) )';
    end
    
    k = k + 1;
    J = norm(VOld - V);
    Js(end+1) = J;
    %
end

% estimate
Xhat = U*V';

% plot
figure;
i = 1;
for k = randperm(n, 8)
    if (i <= 4) subplot(4,4,i);
    else        subplot(4,4,4 + i); end
    imshow(reshape(X(k,:), 50, 50), [])
    title(['Original n°' int2str(k)])
    if (i <= 4) subplot(4,4,4 + i);
    else        subplot(4,4,8 + i); end
    imshow(reshape(Xhat(k,:), 50, 50), [])
    title(['Estimé n°' int2str(k)])
    i = i + 1;
end

figure;
for i = 1:K
    subplot(4,ceil(K/4),i)
    imshow(reshape(V(:,i), 50, 50), [])
    title(['Composante V_{' int2str(i) '}'])
end

figure;
plot(Js)
title('Evolution of V matrix (norm of the variation)');
xlabel('Iteration')
ylabel('Variation norm')
