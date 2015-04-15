function L = oneppv_te(model,X,dummy,Y)

%dataset = prdataset(X, Y);
%L = knn_map(dataset, model);
%L = L.data;

yHat = predict(model, X);

L = zeros(length(Y), max(Y));
for i = 1:max(Y)
    L(yHat == i, i) = 1;
end
