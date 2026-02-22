function simulate_bivariate_mixture_laplace()
n = 10000;
d = 2;
lambda = [0.7, 0.3];
mu1 = [0; 0];
Sigma1 = [1, 0.5; 0.5, 1];
b1 = 10;
mu2 = [0; 0];
Sigma2 = [9, 2; 2, 9];
b2 = 5;

if min(eig(Sigma1)) <= 0 || min(eig(Sigma2)) <= 0
    error('Covariance matrices must be positive definite.');
end

u = rand(n, 1);
C = ones(n, 1);
C(u > lambda(1)) = 2;

Y = zeros(n, d);

idx1 = (C == 1);
n1 = sum(idx1);
G1 = gamrnd(b1, 1, n1, 1);
Z1 = randn(n1, d);
L1 = chol(Sigma1, 'lower');
Y1 = repmat(mu1', n1, 1) + (sqrt(G1) * ones(1, d)) .* (Z1 * L1');

idx2 = (C == 2);
n2 = sum(idx2);
G2 = gamrnd(b2, 1, n2, 1);
Z2 = randn(n2, d);
L2 = chol(Sigma2, 'lower');
Y2 = repmat(mu2', n2, 1) + (sqrt(G2) * ones(1, d)) .* (Z2 * L2');

Y(idx1, :) = Y1;
Y(idx2, :) = Y2;

num_bins = [50 50];
[counts, centers] = hist3(Y, 'Nbins', num_bins);

dx = centers{1}(2) - centers{1}(1);
dy = centers{2}(2) - centers{2}(1);
joint_density = counts / (sum(counts(:)) * dx * dy);

[X_grid, Y_grid] = meshgrid(centers{1}, centers{2});
figure;
surf(X_grid, Y_grid, joint_density', 'EdgeColor', 'none');
xlabel('Y_1', 'FontSize', 12);
ylabel('Y_2', 'FontSize', 12);
zlabel('Estimated Density', 'FontSize', 12);
title('Estimated Joint Density of the Bivariate Mixture Laplace Distribution', 'FontSize', 14);
shading interp;
colormap parula;
colorbar;
view(-30, 45);
end
