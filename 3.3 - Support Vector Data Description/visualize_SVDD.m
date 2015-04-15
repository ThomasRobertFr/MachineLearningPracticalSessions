function []=visualize_SVDD(xi,c,R,pos,col)

set(gcf,'Color',[1,1,1])
hh = plot(xi(:,1),xi(:,2),'+b'); 
set(hh,'LineWidth',2);
hold on
hc = plot(c(1),c(2),'*','Color',col);
set(hc,'LineWidth',2);
h1 = plot(xi(pos,1),xi(pos,2),'ob'); 
set(h1,'LineWidth',1,...
          'MarkerEdgeColor',col,...
          'MarkerSize',15);
t = 0:0.01:2*pi+0.01;
hf = plot(c(1)+sqrt(R)*cos(t),c(2)+sqrt(R)*sin(t),col);
set(hf,'LineWidth',2);
%axis([-3 4 -1 6]);
axis square;