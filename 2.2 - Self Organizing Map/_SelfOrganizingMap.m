%% Couleurs

clc
clear all
close all

color_rgb = [ 0 0 0;
        0 0 255
        173 216 230;
        255 192 203;
        255 0 0;
        255 250 250;
        139 90 43;
        255 255 255;] ;
    
color_id={'black';
'blue1';
'lightblue';
'pink';
'red1';
'snow1';
'tan4';
'white';};

nv=length(color_id);

sD=som_data_struct(color_rgb,'labels',color_id);

figure(1)
sM = som_make(sD,'lattice','rect','msize',[10 10]);
sM = som_autolabel(sM,sD);
som_show(sM, 'color', sM.codebook/255);
% colormap('copper');
hold on
som_grid(sM,'Label',sM.labels,'Labelsize',8,'Line','none','Marker','none','Labelcolor','k');
hold off
% 
% 
figure(2)
sM = som_make(sD,'lattice','hexa','msize',[10 10]);
sM = som_autolabel(sM,sD);
som_show(sM, 'color', sM.codebook/255);
hold on
som_grid(sM,'Label',sM.labels,'Labelsize',8,'Line','none','Marker','none','Labelcolor','k');
hold off

%% Villes de France

%  clear all

temps=dlmread('temps.csv');
villes={'Bordeaux';
'Brest';
'Clermont';
'Grenoble';
'Lille';
'Lyon';
'Marseille';
'Montpellier';
'Nantes';
'Nice';
'Paris';
'Rennes';
'Strasbourg';
'Toulouse';
'Vichy'};

nv=length(villes);

sD=som_data_struct(temps(:,1:12),'labels',villes);

figure(1)
sM = som_make(sD,'lattice','rect','msize',[10 10]);
sM = som_autolabel(sM,sD);
%  som_show(sM,'umat','all');
som_show(sM);
colormap('copper');
hold on
som_grid(sM,'Label',sM.labels,'Labelsize',8,'Line','none','Marker','none','Labelcolor','w');
hold off


figure(2)
sM = som_make(sD,'lattice','hexa','msize',[10 10]);
sM = som_autolabel(sM,sD);
som_show(sM,'umat','all');
colormap('copper');
hold on
som_grid(sM,'Label',sM.labels,'Labelsize',8,'Line','none','Marker','none','Labelcolor','w');
hold off
