function model = oneppv_tr(X, dummy, Y)

%dataset = prdataset(X, Y);
%model = knnc(dataset, 1);

model = fitcknn(X, Y, 'NumNeighbors', 1);


