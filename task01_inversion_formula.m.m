function plot_t_density()
    % Input degree of freedom
    df = input('Enter a positive degree of freedom: ');
    while df <= 0
        df = input('Please enter a positive degree of freedom: ');
    end

    % Define range based on degree of freedom
    if df < 30
        x = linspace(-6, 6, 1000);
    else
        x = linspace(-3, 3, 1000);
    end

    % Calculate exact density
    exact_density = tpdf(x, df);

    % Calculate density using inversion formula
    inversion_density = zeros(size(x));
    for i = 1:length(x)
        inversion_density(i) = inversion_formula(x(i), df);
    end

    % Plotting
    figure;
    plot(x, exact_density, 'b', 'LineWidth', 5.0); hold on;
    plot(x, inversion_density, 'r', 'LineWidth', 2.0);
    xlabel('x');
    ylabel('Density');

    legend('Exact Density', 'Inversion Density');
    grid on;

    % Calculate and plot relative percentage error
    relative_error = (inversion_density - exact_density) ./ exact_density * 100;
    figure;
    plot(x, relative_error, 'k', 'LineWidth', 2.0);
    xlabel('x');
    ylabel('Relative Percentage Error (%)');
    grid on;
    
end

function density = inversion_formula(x, df)
    % Set t_max for integration
    t_max = 100; % Adjust as necessary for convergence
    % Define the integrand function
    integrand = @(t) cos(t * x) .* t_char_function(t, df);
    % Perform the integral
    density = (1 / pi) * integral(integrand, 0, t_max, 'ArrayValued', true);
end

function phi = t_char_function(t, df)
    nu = df / 2;
    z = sqrt(df) * t;
    % Initialize phi
    phi = zeros(size(t));
    
    % Handle t = 0 separately to avoid NaNs
    zero_idx = (t == 0);
    phi(zero_idx) = 1; % Since phi(0) = 1
    
    % Compute phi for t > 0
    pos_idx = ~zero_idx;
    z_pos = z(pos_idx);
    K_nu = besselk(nu, z_pos);
    numerator = (z_pos).^nu .* K_nu;
    denominator = gamma(nu) * 2^(nu - 1);
    phi(pos_idx) = numerator / denominator;
end

