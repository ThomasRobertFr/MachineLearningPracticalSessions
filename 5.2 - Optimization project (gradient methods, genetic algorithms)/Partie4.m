%% Algorithme génétique

clc;
clear all;

% Init
Pxy = [.2 .3 .2 .1 .1 .5 .7 1.1 1.4 1.6
       -.4 .1 1.1 1.1 .9 .7 .9 1.1 1.3 1.5]';

%ti = linspace(0,1,10)';
ti = rand(10, 1);

JOld = Inf;
[param, J] = Estime_a_b_c_d_MC(ti, Pxy);
params{1} = param;
Js = J;

while (abs(JOld - J) > 1e-6)
    JOld = J;

    % Optimize ti
    for i = 1:length(Pxy)
        ti(i) = AlgoGenetique(param, Pxy(i,:)');
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
