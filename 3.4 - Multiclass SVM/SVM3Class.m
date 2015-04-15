function [Xsup, alpha, b] = SVM3Class(Xi, yi, C, kernel, kerneloption, options)

% note yi n'est pas utilisé pour simplifier les calculs. Normalement, il
% devrait influer sur la forme des matrices A, M et Un23.

[n, p] = size(Xi);
ni = n/3;

% matrice A
A = [1 1 -1 0 -1 0 ; -1 0 1 1 0 -1];
A = kron(A,ones(1, ni)) ;

% matrice M et MM
M = [1 -1 0; 1 0 -1 ; -1 1 0 ; 0 1 -1; -1 0 1; 0 -1 1];
MM = kron(M*M', ones(ni));

% matrice Un23
Un23 = [1 0 0;1 0 0 ; 0 1 0 ; 0 1 0; 0 0 1 ; 0 0 1];
Un23 = kron(Un23,eye(ni));

% calcul de G
K = svmkernel(Xi, kernel, kerneloption);
G = MM.*(Un23*K*Un23') ;

% QP
l = 10^ -5;
I = eye(size(G));
G = G + l*I; % kernel matrix
e = ones(2*n,1) ;
[al , ~, pos] = monqp(G, e, A', [0;0], C, l, 0);
n23 = 2*ni;

% calcul de b
yp = G(: ,pos) * al;
b1 = 1 - yp(pos(1)) ;
p2 = (find((pos > n23) & (pos <= 2*n23)));
b2 = 1 - yp(pos(p2(1)));
b3 = 1 - yp(pos(end));

b = [b1; b2; b3];

% calcul de alpha
alpha = zeros(2*n,1);
alpha(pos) = al;

Xsup = Xi;


