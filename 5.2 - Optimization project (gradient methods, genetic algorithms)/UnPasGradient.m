function tiNew = UnPasGradient(param, tiOld, px, py, lambda)

pi = [px;py];

grad = [tiOld^3 tiOld^2 tiOld 1]*(param*param')*[3*tiOld^2; 2*tiOld; 1; 0] ...
      - [3*tiOld^2 2*tiOld 1 0]*param*pi;
grad = 2 * grad;

tiNew = tiOld - lambda * grad;

