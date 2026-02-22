

% Call the simulation and estimation function
%simulate_and_estimate();

 % Call the simulation and estimation function
AIC_matrix = zeros(50,1);
BIC_matrix = zeros(50,1);

for sim = 1:50
  [AIC_matrix(sim,:), BIC_matrix(sim,:)] = simulate_and_estimate();
end


function pdfln = mvnctpdfln(x, mu, gam, v, Sigma)
% x d X T matrix of evaluation points
% mu, gam d-length location and noncentrality vector
% v is df; Sigma is the dispersion matrix.
[d, t] = size(x); C = Sigma; [R, err] = cholcov(C, 0);
assert(err == 0, 'C is not (semi) positive definite');
mu = reshape(mu, length(mu), 1); gam = reshape(gam, length(gam), 1);
vn2 = (v + d) / 2; xm = x - repmat(mu, 1, t); rho = sum((R' \ xm).^2, 1);
pdfln = gammaln(vn2) - d / 2 * log(pi * v) - gammaln(v / 2) - ...
    sum(slog(diag(R))) - vn2 * log1p(rho / v);
if (all(gam == 0)), return; end
idx = (pdfln >= -37); maxiter = 1e4; k = 0;
if (any(idx))
    gcg = sum((R' \ gam).^2); pdfln = pdfln - 0.5 * gcg; xcg = xm' * (C \ gam);
    term = 0.5 * log(2) + log(xcg) - 0.5 * slog(v + rho');
    term(term == -inf) = log(realmin); term(term == +inf) = log(realmax);
    logterms = gammaln((v + d + k) / 2) - gammaln(k + 1) - gammaln(vn2) + k * term;
    ff = real(exp(logterms)); logsumk = log(ff);
    while (k < maxiter)
        k = k + 1;
        logterms = gammaln((v + d + k) / 2) - gammaln(k + 1) - gammaln(vn2) + k * term(idx);
        ff = real(exp(logterms - logsumk(idx))); logsumk(idx) = logsumk(idx) + log1p(ff);
        idx(idx) = (abs(ff) > 1e-4); if (all(idx == false)), break, end
    end
    pdfln = real(pdfln + logsumk');
end
end


function y = slog(x) % Truncated log. No -Inf or +Inf.
y = log(max(realmin, min(realmax, x)));
end




function [param, stderr, iters, loglik, Varcov] = MVNCT2estimation(data)
    [d, T] = size(data);
    assert(d == 2, 'Only supports 2-dimensional data');

    % Define parameter bounds
    bound.lo = [1.1, -30, -30, 0.01, 0.01, -1, -4, -4];
    bound.hi = [100, 30, 30, 100, 100, 1, 4, 4];
    bound.which = [1, 0, 0, 1, 1, 0, 1, 1];
    
    % Define initial guesses
    mu_init = mean(data, 2);             % Empirical mean
    scale_init = std(data, 0, 2);        % Empirical standard deviation
    correlation_init = corr(data(1, :)', data(2, :)'); % Empirical correlation
    noncentrality_init = [0.1, 0.1];     % Small non-zero values
    initvec = [3, mu_init', scale_init', correlation_init, noncentrality_init];

    % Constrain initial guesses using einschraenk
    initvec = einschraenk(initvec, bound);

    % Optimization options
    maxiter = 300;
    tol = 1e-12;
   % opts = optimset('Display', 'iter', 'MaxIter', maxiter, 'TolFun', tol, 'TolX', tol, 'LargeScale', 'off');
       opts = optimset('Display', 'off', 'MaxIter', 300, 'TolFun', 1e-8, ...
                    'TolX', 1e-8, 'MaxFunEvals', 2000, 'Display', 'iter');
    obj_fun = @(param) MVNCTloglik(param, data, bound);

    % Optimize using fminunc
    [pout, fval, ~, output, ~, hess] = fminunc(obj_fun, initvec, opts);

    % Constrain final parameters using einschraenk
    [param, Varcov] = einschraenk(pout, bound, inv(hess) / T);

    % Compute standard errors
    stderr = sqrt(diag(Varcov));

    % Return log-likelihood and iterations
    loglik = -fval * T;
    disp('gammadelta');
    disp(loglik);
    iters = output.iterations;
end



function ll=MVNCTloglik(param,x,bound)
    if nargin<3, bound=0; end
    if isstruct(bound), param=einschraenk(real(param),bound,999); end
    k=param(1); mu=param(2:3); scale=param(4:5); gam=param(7:8);
    R12=param(6); R=[1 R12; R12 1]; 
    if min(eig(R))<1e-4, ll=1e5;
    else
    xx=x; 
    for i=1:2, xx(i,:)=(x(i,:)-mu(i))/scale(i); end
    llvec = mvnctpdfln(xx, mu, gam, k, R) - log(prod(scale));
    ll=-mean(llvec); 
    if isinf(ll), ll=1e5; 
    end
    end
end



function [y, V_out] = einschraenk(x, bound, V_in)
    % Apply parameter constraints
    y = x; % Start with the input vector
    for i = 1:length(x)
        if y(i) < bound.lo(i)
            y(i) = bound.lo(i); % Apply lower bound
        elseif y(i) > bound.hi(i)
            y(i) = bound.hi(i); % Apply upper bound
        end
    end

    % Handle variance-covariance matrix constraints
    if nargin == 3 && ~isempty(V_in)
        V_out = V_in;
        for i = 1:length(y)
            if bound.which(i) == 0
                V_out(i, :) = 0; % Zero out row i
                V_out(:, i) = 0; % Zero out column i
            end
        end
    else
        V_out = []; % Return empty if no input variance-covariance matrix
    end
end

% Simulate data from a multivariate non-central Student's t-distribution
% Simulate data from a multivariate non-central Student's t-distribution
% Simulate data from a multivariate non-central Student's t-distribution
function [AIC, BIC] = simulate_and_estimate()
    % Parameters for the true distribution
    d = 2;                      % Dimension
    T = 8000;                   % Number of samples
    v = 5;                      % Degrees of freedom
    mu = [0; 0];           % Non-centrality parameters
    scale = [1.1, 0.9];         % Standard deviations
    R = [15, 0.3; 0.3, 1];       % Correlation matrix


    % Generate Cholesky decomposition of covariance matrix
    [R_chol, err] = chol(R, 'lower');
    assert(err == 0, 'R is not positive definite');

    % Simulate data
    z = randn(d, T);            % Standard normal samples
    chi2_samples = chi2rnd(v, 1, T);
    x = R_chol * z;             % Apply correlation structure
    x = bsxfun(@times, x, sqrt(v ./ chi2_samples)); % Apply scaling
    x = bsxfun(@plus, x, mu);   % Add non-centrality

    % Scale the data
    for i = 1:d
        x(i, :) = x(i, :) * scale(i);
    end
    %data = x;
    data = DJIA_stock_selection_highest_cov();
    n = size(data, 2); % Number of time points

    % Estimate parameters using provided functions
    [param, stderr, iters, loglik, Varcov] = MVNCT2estimation(data);

    % Display results
    fprintf('Estimated Parameters:\n');
    fprintf('Degrees of Freedom (v): %.4f\n', param(1));
    fprintf('Non-centrality Parameters (mu): %.4f, %.4f\n', param(2), param(3));
    fprintf('Scale Parameters: %.4f, %.4f\n', param(4), param(5));
    fprintf('Correlation: %.4f\n', param(6));
    fprintf('Non-centrality Terms (gamma): %.4f, %.4f\n', param(7), param(8));
    fprintf('Standard Errors:\n');
    disp(stderr);
    fprintf('Log-Likelihood: %.4f\n', loglik);
    fprintf('Iterations: %d\n', iters);

    % True values for comparison
    fprintf('\nTrue Parameters:\n');
    fprintf('Degrees of Freedom (v): %.4f\n', v);
    fprintf('Non-centrality Parameters (mu): %.4f, %.4f\n', mu(1), mu(2));
    fprintf('Scale Parameters: %.4f, %.4f\n', scale(1), scale(2));
    fprintf('Correlation: %.4f\n', R(1, 2));

    disp('loglik');
    disp(loglik);

    num_params_laplace = length(param);
    disp(n);
    % Compute AIC and BIC for each model
    aic_laplace = 2 * num_params_laplace - 2 * loglik;
    bic_laplace = num_params_laplace * log(n) - 2 * loglik;

    AIC = aic_laplace;
    BIC = bic_laplace;

    % Plot sample data and estimated contours
   plot_sample_data_and_contours(data, param);
end

% Function to plot sample data and overlay contours
% Function to plot sample data and overlay contours
% Function to plot sample data and overlay contours
function plot_sample_data_and_contours(x, param)
    % Extract estimated parameters
    v = param(1);
    mu = param(2:3)';
    scale = param(4:5)';
    R12 = param(6);
    R = [1, R12; R12, 1];

    % Generate a grid for contour plot
    % Extend grid to cover more of the data range
    buffer = 0.1; % Fraction of range to extend
    x1_range = linspace(min(x(1, :)) - buffer * range(x(1, :)), max(x(1, :)) + buffer * range(x(1, :)), 200);
    x2_range = linspace(min(x(2, :)) - buffer * range(x(2, :)), max(x(2, :)) + buffer * range(x(2, :)), 200);
    [X, Y] = meshgrid(x1_range, x2_range);
    grid_points = [X(:), Y(:)]';

    % Compute covariance matrix
    Sigma = diag(scale) * R * diag(scale);

    % Compute density at each grid point
    pdf_values = mvnctpdfln(grid_points, mu, param(7:8), v, Sigma);
    pdf_values = reshape(exp(pdf_values), size(X)); % Convert log-PDF to PDF

    % Plot the sample data
    figure;
    scatter(x(1, :), x(2, :), 20, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', [0.6, 0.6, 0.6]);
    hold on;

    % Overlay contours with more levels
    num_contours = 500; % Increase number of contour levels
    contour(X, Y, pdf_values, num_contours, 'LineWidth', 1.5);

    % Plot aesthetics
    title('Sample Data with Estimated Contours');
    xlabel('X1');
    ylabel('X2');
    legend({'Sample Data', 'Estimated Contours'});
    grid on;
    hold off;
end

function dataXX = DJIA_stock_selection_highest_cov()
    fileName = 'DJIA30stockreturns.mat'; % Replace with your .mat file name
    data = load(fileName);
    data = data.DJIARet;
    returns = data;

    % Initialize parameters
    numStocks = size(data, 2); % Number of stocks in the dataset
    numPairs = 50; % Number of stock pairs to analyze
    n = size(data, 1); % Number of time points

    % Validate the dimensions of the dataset
    [num_time_points, num_stocks] = size(returns);
    assert(num_stocks == 25, 'The dataset should contain 25 stocks.');

    % Initialize parameters
    num_iterations = 50; % Number of repetitions (can increase based on resources)
    pairs_selected = zeros(num_iterations, 2); % To store selected stock pairs

    % Compute covariance matrix
    covMatrix = cov(returns);

    % Ensure diagonal is ignored by setting it to -Inf (so it won't be selected as highest)
    covMatrix(logical(eye(size(covMatrix)))) = -Inf;

    % Find indices of all pairs (upper triangle only)
    [row, col] = find(triu(covMatrix, 1)); % Upper triangle indices

    % Calculate probabilities proportional to covariance
    covProbs = covMatrix(sub2ind(size(covMatrix), row, col)); % Extract covariances
    probs = covProbs / sum(covProbs); % Normalize to form a valid probability distribution

    % Randomly select a pair based on the probabilities
    selectedIdx = randsample(length(probs), 1, true, probs);
    pair = [row(selectedIdx), col(selectedIdx)];

    % Extract the data for the selected pair
    pairData = data(:, pair);

    dataX = pairData;
    dataX = dataX.';

    disp('Selected pair with highest covariance:');
    disp(pair);

    dataXX = dataX;
end

