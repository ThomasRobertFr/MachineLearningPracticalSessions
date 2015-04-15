
function datalight = mydownsampling(data, step)

[ndata, dim] = size(data);
assert(step >=1, 'the downsampling step should be greater than 1')

Nlight = ceil(ndata/step);
datalight = zeros(Nlight, dim);
ind = randperm(ndata);
for i=1:Nlight
    datalight(i,:) = data(ind(i), :);
end