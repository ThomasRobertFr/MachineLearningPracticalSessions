%% Optimisation alternée (bi-niveau)

%% ti initialisés linéaire dans [0,1]

clc;
clear all;

% Init
Pxy = [.2 .3 .2 .1 .1 .5 .7 1.1 1.4 1.6
       -.4 .1 1.1 1.1 .9 .7 .9 1.1 1.3 1.5]';
   
ti = linspace(0,1,10)';

JOld = Inf;
[param, J] = Estime_a_b_c_d_MC(ti, Pxy);
params{1} = param;
Js = J;

while (abs(JOld - J) > 1e-9)
    JOld = J;

    % Optimize ti
    for i = 1:length(Pxy)
        ti(i) = optimiseNewton(param, ti(i), Pxy(i,:)');
    end

    % force to meet constrains
    ti(ti < 0) = 0;
    ti(ti > 1) = 1;
    ti = sort(ti);
    
    % Optimize params
    [param, J] = Estime_a_b_c_d_MC(ti, Pxy);
    params{end+1} = param;
    Js(end+1) = J;
end

% Résultats
figure;
tis = 0:0.001:1;
subplot(1,2,1); hold off;
for i = round(linspace(1, length(params), 25))
    mtis = (params{i}'*[tis.^3; tis.^2; tis; ones(size(tis))])';
    plot(mtis(:,1), mtis(:,2), ':k', 'Color', (length(params) - i)/length(params)*[1 1 1]); hold on;
end
plot(mtis(:,1), mtis(:,2), 'LineWidth', 2);
plot(Pxy(:,1), Pxy(:,2), '+r', 'LineWidth', 2);
title('Solution evolution');
subplot(1,2,2); hold off;
plot(Js)
title(['Cost evolution (J_{end} = ' num2str(Js(end)) ')']);
xlabel('Iteration')

%% ti initialisés aléatoirement

clc
clear all

% Init
Pxy = [.2 .3 .2 .1 .1 .5 .7 1.1 1.4 1.6
       -.4 .1 1.1 1.1 .9 .7 .9 1.1 1.3 1.5]';
   
ti = sort(rand(length(Pxy), 1));

param = rand(4,2);
params{1} = param;
JOld = Inf;
J = sumsqr(Pxy - [ti.^3 ti.^2 ti ones(size(ti))]*param);
Js = J;

ite = 1;
while (abs(JOld - J) > 1e-6 && ite < 300)
    JOld = J;

    % Optimize ti
    for i = 1:length(Pxy)
        ti(i) = optimiseNewton(param, ti(i), Pxy(i,:)');
    end

    % force to meet constrains
    ti(ti < 0) = 0;
    ti(ti > 1) = 1;
    ti = sort(ti);
    
    % Optimize params
    [param, J] = Estime_a_b_c_d_MC(ti, Pxy);
    params{end+1} = param;
    Js(end+1) = J;
    
    ite = ite + 1;
end

% Résultats
figure;
tis = 0:0.001:1;
subplot(1,2,1); hold off;
for i = round(linspace(1, length(params), 25))
    mtis = (params{i}'*[tis.^3; tis.^2; tis; ones(size(tis))])';
    plot(mtis(:,1), mtis(:,2), ':k', 'Color', (length(params) - i)/length(params)*[1 1 1]); hold on;
end
plot(mtis(:,1), mtis(:,2), 'LineWidth', 2);
plot(Pxy(:,1), Pxy(:,2), '+r', 'LineWidth', 2);
xlim([-.5, 3])
ylim([-.5, 3])
title('Solution evolution');
subplot(1,2,2); hold off;
plot(Js)
title(['Cost evolution (J_{end} = ' num2str(Js(end)) ')']);
xlabel('Iteration')

%% ti initialisés aléatoire, injection d'aléatoire à chaque convergence

clc
clear all

% Init
Pxy = [.2 .3 .2 .1 .1 .5 .7 1.1 1.4 1.6
       -.4 .1 1.1 1.1 .9 .7 .9 1.1 1.3 1.5]';

ti = sort(rand(length(Pxy), 1));

param = rand(4,2);
params = {};
params{1} = param;

JOld = Inf;
J = sumsqr(Pxy - [ti.^3 ti.^2 ti ones(size(ti))]*param);
Js = J;
plotInds = 1;

for ite = 2:300
    JOld = J;

    % Optimize params
    [param, J] = Estime_a_b_c_d_MC(ti, Pxy);
    params{end+1} = param;
    Js(end+1) = J;
    
    % Optimize ti
    for i = 1:length(Pxy)
        ti(i) = optimiseNewton(param, ti(i), Pxy(i,:)');
    end
    
    % Random if cost does not move a lot
    if (abs(JOld - J) < 1e-3)
        plotInds(end+1) = ite;
        ti = (0.8 + 0.4 * rand(size(ti))) .* ti;
    end
    
    % force to meet constrains
    ti(ti < 0) = 0;
    ti(ti > 1) = 1;
    ti = sort(ti);
end

plotInds(end+1) = ite;

% Résultats
figure;
tis = 0:0.001:1;
subplot(1,2,1); hold off;
for i = plotInds
    mtis = (params{i}'*[tis.^3; tis.^2; tis; ones(size(tis))])';
    plot(mtis(:,1), mtis(:,2), ':k'); hold on;
end
plot(mtis(:,1), mtis(:,2), 'LineWidth', 2);
plot(Pxy(:,1), Pxy(:,2), '+r', 'LineWidth', 2);
xlim([-.5, 3])
ylim([-.5, 3])
title('Solution evolution');
subplot(1,2,2); hold off;
plot(Js)
title(['Cost evolution (J_{end} = ' num2str(Js(end)) ')']);
xlabel('Iteration')

%% ti initialisés aléatoire, injection d'aléatoire à chaque convergence, sauvegarde des meilleurs

clc
clear all

% Init
Pxy = [.2 .3 .2 .1 .1 .5 .7 1.1 1.4 1.6
       -.4 .1 1.1 1.1 .9 .7 .9 1.1 1.3 1.5]';

ti = sort(rand(length(Pxy), 1));

param = rand(4,2);
params = {};
params{1} = param;

JOld = Inf;
J = sumsqr(Pxy - [ti.^3 ti.^2 ti ones(size(ti))]*param);
Js = J;

tibest = ti;
JBest = J;
plotInds = [];

for ite = 2:600
    JOld = J;

    % Optimize params
    [param, J] = Estime_a_b_c_d_MC(ti, Pxy);
    params{ite} = param;
    Js(ite) = J;
    
    % Optimize ti
    for i = 1:length(Pxy)
        ti(i) = optimiseNewton(param, ti(i), Pxy(i,:)');
    end
    
    % Random if cost does not move a lot
    if (abs(JOld - J) < 1e-9)
        if (J < JBest)
            tibest = ti;
            JBest = J;
            itebest = ite;
        end
        plotInds(end+1) = ite;
        ti = (0.8 + 0.4 * rand(size(ti))) .* ti;
    end
    
    % force to meet constrains
    ti(ti < 0) = 0;
    ti(ti > 1) = 1;
    ti = sort(ti);
end

ti = tibest;
[param, J] = Estime_a_b_c_d_MC(ti, Pxy);

% Résultats
figure;
tis = 0:0.001:1;
plotInds(end+1) = itebest;
subplot(1,2,1); hold off;
for i = 1:length(plotInds)
    mtis = (params{plotInds(i)}'*[tis.^3; tis.^2; tis; ones(size(tis))])';
    plot(mtis(:,1), mtis(:,2), ':k'); hold on;
end
plot(mtis(:,1), mtis(:,2), 'LineWidth', 2);
plot(Pxy(:,1), Pxy(:,2), '+r', 'LineWidth', 2);
xlim([-.5, 3])
ylim([-.5, 3])
title('Solution evolution');
subplot(1,2,2); hold off;
plot(Js)
title(['Cost evolution (J_{best} = ' num2str(Js(itebest)) ')']);
xlabel('Iteration')