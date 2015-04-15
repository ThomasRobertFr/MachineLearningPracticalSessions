function H = monHessien(theta)
    H = [(2-400*(theta(2)-theta(1)^2)+800*theta(1)^2) (-400*theta(1)) ; (-400*theta(1)) 200];
end