%% Moindres carrés

clc;
clear all;

% Initialisation
Pxy = [.2 .3 .2 .1 .1 .5 .7 1.1 1.4 1.6
       -.4 .1 1.1 1.1 .9 .7 .9 1.1 1.3 1.5]';
   
ti = linspace(0,1,10)';

% Estimation des paramètres
param = Estime_a_b_c_d_MC(ti, Pxy);

% affichage
figure;
tis = 0:0.001:1;
mtis = (param'*[tis.^3; tis.^2; tis; ones(size(tis))])';

figure;
plot(mtis(:,1), mtis(:,2)); hold on;
plot(Pxy(:,1), Pxy(:,2), '+r');
