%% Gradient & Newton

%% Initilisation des paramètres

clc;
clear all;

param = [4.2 12.3
         -3.2 -20.2
         0.4 9.9
         0.2 -0.4];

px = 0.1;
py = 0.9;
pi = [px ; py];

lambda = 0.02;

%% Gradient & Newton - Essais de 4 itérations

% Essai de plusieurs valeurs de ti
for tiInit = [1 0.02]

    % Gradient
    
    tiOld = tiInit;
    
    mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
    JOld = sumsqr(pi - mti);
    JNew = Inf;

    for i = 1:4
        tiNew = UnPasGradient(param, tiOld, px, py, lambda);
        mti = param'*[tiNew^3; tiNew^2; tiNew; 1];
        JNew = sumsqr(pi - mti);

        if (JNew > JOld)
            lambda = lambda / 2;
        else
            lambda = lambda * 1.5;
            tiOld = tiNew;
            JOld = JNew;
        end
    end

    fprintf('Gradient | tiInit : %.3f | tiFinal : %.3f | cout : %.6f\n', tiInit, tiOld, JOld);

    % Newton

    tiOld = tiInit;

    for i = 1:4
        tiOld = UnPasGaussNewton(param, tiOld, px, py);
        mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
        JOld = sumsqr(pi - mti);
    end
    
    fprintf('Newton   | tiInit : %.3f | tiFinal : %.3f | cout : %.6f\n', tiInit, tiOld, JOld);
    
end

%% Gradient & Newton - Etude

% Initialisation des elements
tis = 0:0.001:1;
n = length(tis);
mtis = (param'*[tis.^3; tis.^2; tis; ones(1, n)])';
Js = sum((repmat(pi', n, 1) - mtis).^2, 2);

% Initialisation de la figure
figure(1);
subplot(2,2,1); hold off;
plot(mtis(:,1), mtis(:,2)); hold on;
plot(pi(1), pi(2), 'c+');
title('m(t) - gradient');
subplot(2,2,3); hold off;
plot(mtis(:,1), mtis(:,2)); hold on;
plot(pi(1), pi(2), 'c+');
title('m(t) - Gauss-Newton');

subplot(2,2,2); hold off;
plot(tis, Js);  hold on;
title('J - gradient');
subplot(2,2,4); hold off;
plot(tis, Js);  hold on;
title('J - Gauss-Newton');

% Gradient 1

tiOld = 0.1;
lambda = 0.005;

mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
JOld = sumsqr(pi - mti);
JNew = Inf;

subplot(2,2,1);
plot(mti(1), mti(2), 'g+');
subplot(2,2,2);
plot(tiOld, JOld, 'g+');

for i = 1:4
    tiNew = UnPasGradient(param, tiOld, px, py, lambda);
    mti = param'*[tiNew^3; tiNew^2; tiNew; 1];
    JNew = sumsqr(pi - mti);

    if (JNew > JOld)
        lambda = lambda / 2;
    else
        lambda = lambda * 1.5;
        tiOld = tiNew;
        JOld = JNew;
        
        subplot(2,2,1);
        plot(mti(1), mti(2), 'g+');
        subplot(2,2,2);
        plot(tiOld, JOld, 'g+');
    end
end

% Gradient 2

tiOld = 0.9;
lambda = 0.005;

mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
JOld = sumsqr(pi - mti);
JNew = Inf;

subplot(2,2,1);
plot(mti(1), mti(2), 'r+');
subplot(2,2,2);
plot(tiOld, JOld, 'r+');

for i = 1:8
    tiNew = UnPasGradient(param, tiOld, px, py, lambda);
    mti = param'*[tiNew^3; tiNew^2; tiNew; 1];
    JNew = sumsqr(pi - mti);

    if (JNew > JOld)
        lambda = lambda / 2;
    else
        lambda = lambda * 1.5;
        tiOld = tiNew;
        JOld = JNew;
        
        subplot(2,2,1);
        plot(mti(1), mti(2), 'r+');
        subplot(2,2,2);
        plot(tiOld, JOld, 'r+');
    end
end

% Newton 1

tiOld = 0.1;

mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
JOld = sumsqr(pi - mti);
JNew = Inf;

subplot(2,2,3);
plot(mti(1), mti(2), 'g+');
subplot(2,2,4);
plot(tiOld, JOld, 'g+');

for i = 1:4
    tiOld = UnPasGaussNewton(param, tiOld, px, py);
    mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
    JOld = sumsqr(pi - mti);
    
    subplot(2,2,3);
    plot(mti(1), mti(2), 'g+');
    subplot(2,2,4);
    plot(tiOld, JOld, 'g+');
end

% Newton 2

tiOld = 0.9;

mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
JOld = sumsqr(pi - mti);
JNew = Inf;

subplot(2,2,3);
plot(mti(1), mti(2), 'r+');
subplot(2,2,4);
plot(tiOld, JOld, 'r+');

for i = 1:4
    tiOld = UnPasGaussNewton(param, tiOld, px, py);
    mti = param'*[tiOld^3; tiOld^2; tiOld; 1];
    JOld = sumsqr(pi - mti);
    
    subplot(2,2,3);
    plot(mti(1), mti(2), 'r+');
    subplot(2,2,4);
    plot(tiOld, JOld, 'r+');
end





