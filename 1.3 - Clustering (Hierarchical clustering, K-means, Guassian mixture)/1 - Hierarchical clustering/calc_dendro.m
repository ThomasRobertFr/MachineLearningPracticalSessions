function [M, level, level_single] = calc_dendro(data, show)

M = distance(data, 'euclid');
M = M + diag(inf*ones(1,size(M)));

level = aggclust(M, 'complete');
level_single = aggclust(M, 'single');

if (show)
    subplot(1,2,1);
    dendro(level);
    title('Dendrogamme avec la méthode complete');

    subplot(1,2,2);
    dendro(level_single);
    title('Dendrogamme avec la méthode single');
end