function level = aggclust(distance, method)
% AGGCLUST Hierarchical (agglomerative) clustering
%	Usage: level = aggclust(distance, method)
%
%	distance: 2D distance matrix of data points, with diagonal elements
%		of "INF"
%	method: "single" for single-linkage
%		"complete" for complete-linkage
%	level: data structure for a hierarchical clustering result
%	level(i).distance: distance matrix at level i
%	level(i).height: the minimum distance measure to form level i 
%	level(i).merged: the two clusters (of level i-1) being merged to form
%		level i 
%	level(i).cluster{j}: a vector denotes the data points in j-th cluster
%		of level i 
%
%	Type "aggclust" to see a demo of a hierarchical clustering
%	(single-linkage) of 50 random patterns of dimensionality 2.
%
%	See also DENDRO, LINKCLU.

% Roger Jang, 981027

if nargin == 0, selfdemo; return; end
if nargin < 2, method = 'complete'; end

data_n = size(distance, 1);
level(1).distance = distance;
level(1).height = 0;
level(1).merged = [];
for i = 1:data_n,
	level(1).cluster{i} = [i];
end

for i = 2:data_n,
	level(i) = merge(level(i-1), method);
end

% ====== Merge clusters
function level_out = merge(level, method)
% MERGE Merge a level of n clusters into n-1 clusters

cluster_n = length(level.cluster);
[min_i, min_j, min_value] = minxy(level.distance);
% Reorder to have min_i < min_j
if min_i > min_j,
	temp = min_i;
	min_i = min_j; 
	min_j = temp;
end

level_out = level;

% Update height
level_out.height = min_value;

% Update merged cluster
level_out.merged = [min_i min_j];

% Update cluster
level_out.cluster{min_i} = [level_out.cluster{min_i} level_out.cluster{min_j}];
level_out.cluster(min_j) = [];	% delete cluster{min_j}

% New distance matrix
distance2 = level.distance;
% "min" for single-linkage; "max" for complete-linkage
if strcmp(method, 'single'),
	distance2(:, min_i) = min(distance2(:, min_i), distance2(:, min_j)); 
	distance2(min_i, :) = min(distance2(min_i, :), distance2(min_j, :)); 
elseif strcmp(method, 'complete'),
	distance2(:, min_i) = max(distance2(:, min_i), distance2(:, min_j)); 
	distance2(min_i, :) = max(distance2(min_i, :), distance2(min_j, :)); 
else
	error('Unsupported method in AGGCLUST!');
end

distance2(min_j, :) = [];
distance2(:, min_j) = [];
distance2(min_i, min_i) = inf;

level_out.distance = distance2;

% ====== Find the minimum value in a matrix
function [i, j, min_value] = minxy(A)
[value_row, index_row] = min(A);
[min_value, j] = min(value_row);
i = index_row(j);

% ====== Self demo ======
function selfdemo
data_n = 50;
dimension = 2;
points = rand(data_n, dimension);
for i = 1:data_n,
	for j = 1:data_n,
		distance(i, j) = norm(points(i,:)-points(j,:));
	end
end

% Diagonal elements should always be inf.
for i = 1:data_n, distance(i, i) = inf; end

level = aggclust(distance);

% Plot heights w.r.t. levels
figure;
plot([level.height], 'r:o');
xlabel('Level');
ylabel('Height');
title('Height vs. level');

% Plot the dendrogram
figure;
dendro(level);

% View the formation of clusters
figure;
linkclu(points, distance, level);
