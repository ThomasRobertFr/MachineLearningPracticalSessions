function ti = optimiseNewton(param, ti, pi)

mti = param'*[ti^3; ti^2; ti; 1];

JOld = Inf;
JNew = sumsqr(pi - mti);

while (abs(JOld - JNew) > 1e-6)
    JOld = JNew;
    ti = UnPasGaussNewton(param, ti, pi(1), pi(2));
    mti = param'*[ti^3; ti^2; ti; 1];
    JNew = sumsqr(pi - mti);
end
