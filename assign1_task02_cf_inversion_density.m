function plot_sum_t_distributions_two_simulations()
    % Input degrees of freedom
    df1 = input('Enter the first positive degree of freedom (e.g., 3): ');
    while df1 <= 0
        df1 = input('Please enter a positive degree of freedom: ');
    end

    df2 = input('Enter the second positive degree of freedom (e.g., 5): ');
    while df2 <= 0
        df2 = input('Please enter a positive degree of freedom: ');
    end

    % Define range for z based on degrees of freedom
    if df1 < 30 || df2 < 30
        z = linspace(-10, 10, 1000);
    else
        z = linspace(-6, 6, 1000);
    end

    % Calculate convolution density using quadgk
    conv_density = zeros(size(z));
    for i = 1:length(z)
        conv_density(i) = convolution_density(z(i), df1, df2);
    end

    % Calculate normal approximation density
    [normal_density, total_var] = normal_approx_density(z, df1, df2);

    % Calculate density using characteristic function inversion
    cf_density = cf_inversion_density(z, df1, df2);

    % Simulate densities with different simulation sizes
    N_large = 1e6; % Simulation size for first plot
    N_small = 1e4; % Simulation size for second plot

    % Simulated density with N_large
    sim_density_large = simulate_sum_density(z, df1, df2, N_large);

    % Simulated density with N_small
    sim_density_small = simulate_sum_density(z, df1, df2, N_small);

    % First plot with N_large
    figure;
    plot(z, conv_density, 'b', 'LineWidth', 2.0); hold on;
    plot(z, sim_density_large, 'k-.', 'LineWidth', 2.0);
    plot(z, cf_density, 'm:', 'LineWidth', 2.0);
    xlabel('z', 'FontSize', 12);
    ylabel('Density', 'FontSize', 12);
    legend('Convolution Density', 'Simulated Density', 'CF Inversion Density', 'Location', 'Best');
    grid on;

    
    % Second plot with N_small
    figure;
    plot(z, conv_density, 'b', 'LineWidth', 2.0); hold on;
    plot(z, sim_density_small, 'k-.', 'LineWidth', 2.0);
    plot(z, cf_density, 'm:', 'LineWidth', 2.0);
    xlabel('z', 'FontSize', 12);
    ylabel('Density', 'FontSize', 12);
    legend('Convolution Density', 'Simulated Density', 'CF Inversion Density', 'Location', 'Best');
    grid on;


    % Adjust figure size as needed
    set(gcf, 'Position', [100, 100, 1200, 500]); 
end

function density = convolution_density(z, df1, df2)
    % Define the integrand function for convolution
    integrand = @(x) tpdf(x, df1) .* tpdf(z - x, df2);
    % Perform the integral over the real line using quadgk
    density = quadgk(integrand, -Inf, Inf, 'RelTol', 1e-6, 'AbsTol', 1e-9);
end

function [density, total_var] = normal_approx_density(z, df1, df2)
    % Variance of each t-distribution
    if df1 > 2
        var1 = df1 / (df1 - 2);
    else
        var1 = Inf; % Variance undefined for df <= 2
    end

    if df2 > 2
        var2 = df2 / (df2 - 2);
    else
        var2 = Inf;
    end

    % Total variance
    total_var = var1 + var2;

    if isfinite(total_var)
        % Normal approximation density
        density = normpdf(z, 0, sqrt(total_var));
    else
        % Variance is infinite; normal approximation not valid
        density = zeros(size(z));
    end
end

function density = simulate_sum_density(z, df1, df2, N)
    % Simulate two independent t-distributed random variables
    rng('default'); % For reproducibility
    X = trnd(df1, N, 1); % Simulate from t-distribution with df1
    Y = trnd(df2, N, 1); % Simulate from t-distribution with df2

    % Compute the sum
    Z = X + Y;

    % Estimate the density using ksdensity
    [density_values, density_points] = ksdensity(Z, z, 'Function', 'pdf', 'Bandwidth', 0.1);

    % Since ksdensity may not return density at all z-points, interpolate
    density = interp1(density_points, density_values, z, 'linear', 0);
end

function density = cf_inversion_density(z, df1, df2)
    % Set t_max for integration
    t_max = 100; % Adjust as necessary for convergence
    % Define t values for integration
    N_t = 10000; % Number of points in t-space
    t = linspace(-t_max, t_max, N_t);

    % Compute the combined characteristic function
    phi_combined = t_char_function(t, df1) .* t_char_function(t, df2);

    % Compute the inverse Fourier transform
    density = zeros(size(z));
    for i = 1:length(z)
        x = z(i);
        integrand = phi_combined .* exp(-1i * t * x);
        density(i) = (1 / (2 * pi)) * trapz(t, integrand);
    end

    % Take the real part (density should be real-valued)
    density = real(density);
end

function phi = t_char_function(t, df)
    nu = df / 2;
    z = sqrt(df) * abs(t);
    % Initialize phi
    phi = zeros(size(t));

    % Handle t = 0 separately to avoid NaNs
    zero_idx = (t == 0);
    phi(zero_idx) = 1; % Since phi(0) = 1

    % Compute phi for t ~= 0
    pos_idx = ~zero_idx;
    z_pos = z(pos_idx);
    K_nu = besselk(nu, z_pos);
    numerator = (z_pos).^nu .* K_nu;
    denominator = gamma(nu) * 2^(nu - 1);
    phi(pos_idx) = numerator ./ denominator;

    % Since t-distribution is symmetric, phi(t) is real-valued
    phi = phi .* exp(1i * t * 0); % Include location parameter if needed
end
