function tiNew = UnPasGaussNewton(param, tiOld, px, py)

pi = [px ; py];

grad = [tiOld^3 tiOld^2 tiOld 1]*(param*param')*[3*tiOld^2; 2*tiOld; 1; 0] ...
      - [3*tiOld^2 2*tiOld 1 0]*param*pi;
grad = 2 * grad;

H = [tiOld^3 tiOld^2 tiOld 1]*(param*param')*[6*tiOld; 2; 0; 0] ...
      + [3*tiOld^2 2*tiOld 1 0]*(param*param')*[3*tiOld^2; 2*tiOld; 1; 0] ...
      - [6*tiOld 2 0 0]*param*pi;
H = 2 * H;

tiNew = tiOld - H \ grad;

