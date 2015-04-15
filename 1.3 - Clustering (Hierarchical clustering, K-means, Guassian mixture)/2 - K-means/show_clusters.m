function show_clusters(X, liste)

clusters = unique(liste);
nbClust = length(clusters);

colors = hsv(nbClust);

for i = 1:nbClust

    dataClust = X(liste == clusters(i), :);
    hold on;
    plot(dataClust(:,1), dataClust(:,2), '.', 'color', colors(i, :));
    
end