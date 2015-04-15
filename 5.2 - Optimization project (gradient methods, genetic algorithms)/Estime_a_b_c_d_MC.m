function [param, crit] = Estime_a_b_c_d_MC(ti, Pxy)

theta = [ti.^3 ti.^2 ti ones(size(ti))];

param = (theta'*theta)\(theta'*Pxy);

mtis = (theta*param);

crit = sumsqr(Pxy - mtis);
