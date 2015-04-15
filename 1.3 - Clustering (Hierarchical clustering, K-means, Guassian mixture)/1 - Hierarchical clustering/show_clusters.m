function show_clusters(data, level, nbClust)

colors = hsv(nbClust);

for i = 1:nbClust

    inds = level(end - nbClust + 1).cluster{i};
    dataClust = data(inds, :);
    hold on;
    plot(dataClust(:,1), dataClust(:,2), '*', 'color', colors(i, :));
    
end