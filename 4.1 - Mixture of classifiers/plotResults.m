function plotResults(results, ytitle, legends, colors, styles, tops)

nbTops = size(results, 1);
nbCl = size(results, 2);

if (nargin < 2)
    ytitle = 'classification';
end

if (nargin < 3)
    legends = strcat(cellfun(@(x) 'Cl', num2cell(1:nbCl), 'UniformOutput', false), cellfun(@num2str, num2cell(1:nbCl), 'UniformOutput', false));
end

if (nargin < 4)
    colors = {'b', 'r', 'g', 'm', 'k'};
end

if (nargin < 5)
    styles = {'-+', '-x'};
end

if (nargin < 6)
    tops = strcat(cellfun(@(x) 'Top ', num2cell(1:nbTops), 'UniformOutput', false), cellfun(@num2str, num2cell(1:nbTops), 'UniformOutput', false));
else
    tops = strcat(cellfun(@(x) 'Top ', num2cell(tops), 'UniformOutput', false), cellfun(@num2str, num2cell(tops), 'UniformOutput', false));
end

for i = 1:size(results,2)
    plot(results(:,i)*100, [styles{mod(i+1,length(styles)) + 1} colors{mod(ceil(i/length(styles))-1,length(colors)) + 1}]);
    hold on;
end
set(gca, 'XTick', 1:nbTops);
set(gca, 'XTickLabel', tops);
if (~isempty(legends))
    legend(legends, 'Location', 'Best');
end
ylabel(['Taux de ' ytitle ' (%)']);
