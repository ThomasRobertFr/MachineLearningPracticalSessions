%% Prepare data
% 
% First we prepare the data: we standardize the data and create 2 datasets
% for learning and testing.

clear all;
close all;
data = load('housing.data');

% make X and y matrices
[n,d] = size(data);
p = d-1;
X = data(:, 1:p);
y = data(:,d);   

% standardize feature values and center target
mu_y = mean(y);
y = y - mu_y;
[X, mu, sigma] = standardizeCols(X);

% Split learn and test
[Xlearn, ylearn, Xtest, ytest] = splitdata(X, y, 0.5);

% Rename elements because I'm lazy
X = Xlearn;
y = ylearn;
n = length(y);

%% Compute lasso

% Hyperparameter

lambda = 10;

% Standard lasso

bLasso = CWLasso(X, y, lambda, zeros(p,1));

% Primal Adapatative Lasso with CVX

w = 1./abs((X'*X)\(X'*y));

cvx_quiet(true);
cvx_begin
    variables b(p)
    minimise(1/2 * sum_square(y - X * b) + lambda * w' * abs(b))
cvx_end

bPrimCVX = b;

% Primal Adapatative Lasso with monQP

k = w'*abs(bPrimCVX);

A = [w
     w];
c = [ X'*y
     -X'*y];
b = k;
H = [ X'*X -X'*X
     -X'*X  X'*X];
C = ones(2*p,1)*Inf;

[bvals, ~, pos] = monqp(H, c, A, b, C, 1e-6, false);
b = zeros(2*p, 1);
b(pos) = bvals;
b = b(1:p) - b(p+1:end);

bPrimQP = b;

% Dual Adaptative Lasso with CVX

cvx_begin
    variables a(n)
    minimise(1/2 * sum_square(a) - a'*y)
    subject to
        abs(X'*a) <= lambda*w
cvx_end

bDualCVX1 = (X'*X)\(X'*(y - a))

% Dual Adaptative Lasso with CVX v2

cvx_begin
    variables b(p)
    minimise(1/2 * sum_square(X * b))
    subject to
        abs(X'*(y-X*b)) <= lambda*w
cvx_end

bDualCVX2 = b

% Plot results

[bLasso bPrimCVX bPrimQP bDualCVX1 bDualCVX2]

