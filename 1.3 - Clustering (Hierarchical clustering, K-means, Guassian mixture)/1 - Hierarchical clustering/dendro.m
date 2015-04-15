function dendro(level)
% DENDRO Dendrogrma plot for the result from hierarchical clustering.
%
%	Usage: dendro(level)
%	level: data structure for a hierarchical clustering result
%	level(i).distance: distance matrix at level i
%	level(i).height: the minimum distance measure to form level i 
%	level(i).merged: the two clusters (of level i-1) being merged to form
%		level i 
%	level(i).cluster{j}: a vector denotes the data points in j-th cluster
%		of level i 
%
%	Type "dendro" to see a demo of a hierarchical clustering
%	(single-linkage) of 50 random patterns of dimensionality 2.
%
%	See also AGGCLUST, HCLUSTDM.

%	Roger Jang, 981027

if nargin == 0, selfdemo; return, end

set(gca, 'xticklabel', []);
xticklabel = level(end).cluster{1};
data_n = length(level);
axis([1, data_n, 0, level(end).height]); 
xlabel('Data index');
ylabel('Distance');
title('Dendrogram');
for i=1:data_n,
	h = text(i, 0, num2str(level(end).cluster{1}(i)));
	set(h, 'rot', 90, 'fontsize', 8, 'hori', 'right');
end

% Generate necessary information for plotting dendrogram
% cap_center is the leg position for future cluster
cap_center(xticklabel) = 1:data_n;
levelinfo(1).cap_center = cap_center; 
% cap_height is the height for each cap
levelinfo(1).cap_height = zeros(1, data_n);
for i = 2:data_n,
	m = level(i).merged(1);
	n = level(i).merged(2);
	% Find cap_center
	levelinfo(i).cap_center = levelinfo(i-1).cap_center;
	levelinfo(i).cap_center(m) = ...
		(levelinfo(i).cap_center(m)+levelinfo(i).cap_center(n))/2; 
	levelinfo(i).cap_center(n) = [];
	% Find cap_height
	levelinfo(i).cap_height = levelinfo(i-1).cap_height;
	levelinfo(i).cap_height(m) = level(i).height;
	levelinfo(i).cap_height(n) = [];
end

% Plot caps for the dendrogram
center = 1:data_n;	% center for each cluster
for i = 2:data_n,
	height = level(i).height;
	m = level(i).merged(1);
	n = level(i).merged(2);
	cluster1 = level(i-1).cluster{m};
	cluster2 = level(i-1).cluster{n};
	left_point = cluster1(end);
	right_point = cluster2(1);
	left = find(xticklabel==left_point);
	right = find(xticklabel==right_point);

	left_x = levelinfo(i-1).cap_center(m);
	left_y = levelinfo(i-1).cap_height(m);
	right_x = levelinfo(i-1).cap_center(n);
	right_y = levelinfo(i-1).cap_height(n);
	line([left_x left_x], [left_y, height]);
	line([right_x right_x], [right_y, height]);
	line([left_x right_x], [height, height]);
end

% Plot level lines for the dendrogram
%for i = 1:data_n,
%	line([1 data_n], [level(i).height level(i).height], ...
%		'color', 'c', 'linestyle', ':');
%end

% ====== Self demo ======
function selfdemo
data_n = 50;
dimension = 2;
points = rand(data_n, dimension);
% Compute the distance matrix
for i = 1:data_n,
	for j = 1:data_n,
		distance(i, j) = norm(points(i,:)-points(j,:));
	end
end
% Diagonal elements should always be inf.
for i = 1:data_n, distance(i, i) = inf; end

level = aggclust(distance);

% Plot the dendrogram
figure;
dendro(level);
fprintf('The figure is a dendrogram (single-linkage) of 50 random points in 2D\n');

