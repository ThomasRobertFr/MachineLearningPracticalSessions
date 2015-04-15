function init_fig(theta0, Jmat, n, X, Y)

    % Create the surface plot using contour command
    figure;
    contour(X, Y, reshape(Jmat, n, n), 10, 'linewidth', 1.5);
    colorbar;

    % trace de theta0
    hold on
    h = plot(theta0(1,:), theta0(2,:), 'ro');
    set(h, 'MarkerSize', 8, 'markerfacecolor', 'r');
    text(theta0(1), theta0(2)+0.025, '\theta_0', 'fontsize', 15)

