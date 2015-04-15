function showTable(results, colonnes, lignes)

f = figure;
t = uitable('Data', results, 'ColumnName', colonnes, 'RowName', lignes);

% change sizes
tableextent = get(t,'Extent');
oldposition = get(t,'Position');
newposition = [oldposition(1) oldposition(2) tableextent(3) tableextent(4)];
set(t, 'Position', newposition);

padding = 50;
oldposition = get(f,'Position');
newposition = [oldposition(1) oldposition(2) tableextent(3) + padding tableextent(4) + padding];
set(f, 'Position', newposition);
