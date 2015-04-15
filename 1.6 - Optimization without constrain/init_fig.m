function init_fig(theta0, Jmat, n, X, Y)

    % Create the surface plot using contour command
    figure;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperPosition', [0 0 750*1.7 500*1.7]);
    set(gcf, 'Position', [0 0 750*1.7 500*1.7]);
    contour(X, Y, reshape(Jmat, n, n), [40:-5:0 1 0], 'linewidth', 1.5);
    colorbar
    axis tight

    % trace de theta0
    hold on
    h = plot(theta0(1,:), theta0(2,:), 'ro');
    set(h, 'MarkerSize', 8, 'markerfacecolor', 'r');
    text(theta0(1), theta0(2)+0.025, '\theta_0', 'fontsize', 15)

