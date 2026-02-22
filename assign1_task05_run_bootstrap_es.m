% Parameters
df = 5;
n = 1000;
B = 500;
ESlevel = 0.05;

% Call the function
[ES_true, ES_sample, ES_bootstrap_mean, ES_bootstrap_std, bootstrap_CI] = compute_ES_bootstrap(df, n, B, ESlevel);

% Main code
nu = 5;  % Degrees of freedom
% Compute ES with default location, scale, and ESlevel (0.05)
ES = compute_ES_student_t(nu);

% Display the result
fprintf('Expected Shortfall (ES) with nu = %d: %.4f\n', nu, ES);


function [ES_true, ES_sample, ES_bootstrap_mean, ES_bootstrap_std, bootstrap_CI] = compute_ES_bootstrap(df, n, B, ESlevel)
    % compute_ES_bootstrap computes the true ES, sample ES, and bootstrap estimates of ES.
    % It also generates plots to visualize the data distribution and bootstrap ES estimates.
    %
    % Inputs:
    %   df       - Degrees of freedom (must be > 1)
    %   n        - Sample size (default: 500)
    %   B        - Number of bootstrap replications (default: 500)
    %   ESlevel  - Tail probability (default: 0.05)
    %
    % Outputs:
    %   ES_true           - The true Expected Shortfall value for the specified parameters
    %   ES_sample         - The sample Expected Shortfall calculated from the generated data
    %   ES_bootstrap_mean - Mean of the bootstrap ES estimates
    %   ES_bootstrap_std  - Standard deviation (standard error) of the bootstrap ES estimates
    %   bootstrap_CI      - Confidence interval for ES based on bootstrap (e.g., 95% CI)
    %
    % Example Usage:
    %   [ES_true, ES_sample, ES_bootstrap_mean, ES_bootstrap_std, bootstrap_CI] = compute_ES_bootstrap(4, 250, 500, 0.05);
    
    % -------------------------------
    % 1. Input Handling and Defaults
    % -------------------------------
    
    % Set default values if inputs are not provided or empty
    if nargin < 4 || isempty(ESlevel)
        ESlevel = 0.05;
    end
    if nargin < 3 || isempty(B)
        B = 500;
    end
    if nargin < 2 || isempty(n)
        n = 500;
    end
    
    % Validate inputs
    if df <= 1
        error('Degrees of freedom (df) must be greater than 1 to ensure existence of ES.');
    end
    if n <= 0 || floor(n) ~= n
        error('Sample size n must be a positive integer.');
    end
    if B <= 0 || floor(B) ~= B
        error('Number of bootstrap replications B must be a positive integer.');
    end
    if ESlevel <= 0 || ESlevel >= 1
        error('ESlevel must be between 0 and 1 (exclusive).');
    end
    
    % -------------------------------
    % 2. Data Generation
    % -------------------------------
    
    % Generate random data set of IID Student's t realizations
    data = trnd(df, n, 1); % location=0, scale=1
    
    % -------------------------------
    % 3. True Expected Shortfall Calculation
    % -------------------------------
    
    % Compute the true ES using the function compute_ES_student_t
    ES_true = compute_ES_student_t(df, 0, 1, ESlevel);
    
    % -------------------------------
    % 4. Sample Expected Shortfall Calculation
    % -------------------------------
    
    % Compute the sample VaR (Value at Risk) at the ESlevel
    VaR_sample = quantile(data, ESlevel);
    
    % Compute the sample ES as the mean of observations <= VaR_sample
    ES_sample = mean(data(data <= VaR_sample));
    
    % -------------------------------
    % 5. Bootstrap Resampling
    % -------------------------------
    
    % Initialize array to store bootstrap ES estimates
    ES_bootstrap = zeros(B, 1);
    
    % Perform bootstrap resampling
    fprintf('Performing bootstrap resampling (%d replications)...\n', B);
    parfor b = 1:B
        % Resample data with replacement
        bootstrap_sample = data(randi(n, n, 1));
        
        % Compute bootstrap sample VaR and ES
        VaR_bootstrap = quantile(bootstrap_sample, ESlevel);
        ES_bootstrap(b) = mean(bootstrap_sample(bootstrap_sample <= VaR_bootstrap));
    end
    
    % -------------------------------
    % 6. Bootstrap Statistics
    % -------------------------------
    
    % Compute bootstrap statistics
    ES_bootstrap_mean = mean(ES_bootstrap);
    ES_bootstrap_std = std(ES_bootstrap);
    
    % Compute bootstrap confidence interval (e.g., 95% CI)
    alpha_CI = 0.05; % For 95% CI
    lower_percentile = 100 * (alpha_CI / 2);
    upper_percentile = 100 * (1 - alpha_CI / 2);
    bootstrap_CI = prctile(ES_bootstrap, [lower_percentile, upper_percentile]);
    
    % -------------------------------
    % 7. Display the Results
    % -------------------------------
    
    fprintf('\n--- Expected Shortfall (ES) Analysis ---\n');
    fprintf('Degrees of Freedom (df): %d\n', df);
    fprintf('Sample Size (n): %d\n', n);
    fprintf('Number of Bootstrap Replications (B): %d\n', B);
    fprintf('ES Level (ESlevel): %.2f\n\n', ESlevel);
    
    fprintf('True Expected Shortfall (ES): %.4f\n', ES_true);
    fprintf('Sample Expected Shortfall (ES): %.4f\n', ES_sample);
    fprintf('Bootstrap Mean ES: %.4f\n', ES_bootstrap_mean);
    fprintf('Bootstrap Standard Error of ES: %.4f\n', ES_bootstrap_std);
    fprintf('Bootstrap %.1f%% Confidence Interval for ES: [%.4f, %.4f]\n', 100*(1-alpha_CI), bootstrap_CI(1), bootstrap_CI(2));
    
    % -------------------------------
    % 8. Plotting
    % -------------------------------
    
    % Create a figure with two subplots
    figure('Color', 'w', 'Position', [100, 100, 1200, 600]);
    
    % -------------------------------
    % 8a. Plot 1: Data Distribution with VaR and ES
    % -------------------------------
    
    subplot(1, 2, 1);
    % Histogram of data
    histogram(data, 'Normalization', 'pdf', 'BinMethod', 'sturges', 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    
    % Plot the PDF of the t-distribution
    x_values = linspace(min(data)-1, max(data)+1, 1000);
    pdf_t = tpdf(x_values, df);
    plot(x_values, pdf_t, 'b-', 'LineWidth', 2);
    
    % Plot VaR line
    y_VaR = tpdf(VaR_sample, df);
    plot([VaR_sample, VaR_sample], [0, y_VaR], 'r--', 'LineWidth', 2);
    
    % Plot ES line
    y_ES = tpdf(ES_sample, df);
    plot([ES_sample, ES_sample], [0, y_ES], 'g--', 'LineWidth', 2);
    
    % Shade the area beyond VaR (left tail)
    x_fill = linspace(min(data), VaR_sample, 100);
    y_fill = tpdf(x_fill, df);
    fill([x_fill, VaR_sample], [y_fill, 0], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Annotations
    text(VaR_sample, y_VaR*1.05, sprintf(' VaR = %.2f', VaR_sample), 'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    text(ES_sample, y_ES*1.05, sprintf(' ES = %.2f', ES_sample), 'FontSize', 12, 'Color', 'g', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    
    % Labels and Title
    xlabel('Loss', 'FontSize', 14);
    ylabel('Probability Density', 'FontSize', 14);
    title('Data Distribution with VaR and ES', 'FontSize', 16);
    
    % Legend
    legend('Histogram of Data', 't-Distribution PDF', 'VaR Threshold', 'ES Threshold', 'VaR Region', 'Location', 'Best');
    
    grid on;
    box on;
    
    % -------------------------------
    % 8b. Plot 2: Bootstrap ES Estimates
    % -------------------------------
    
    subplot(1, 2, 2);
    % Histogram of bootstrap ES estimates
    histogram(ES_bootstrap, 'Normalization', 'pdf', 'BinMethod', 'sturges', 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    
    % Plot density of bootstrap ES
    [f_bootstrap, xi_bootstrap] = ksdensity(ES_bootstrap);
    plot(xi_bootstrap, f_bootstrap, 'b-', 'LineWidth', 2);
    
    % Plot mean of bootstrap ES
    plot([ES_bootstrap_mean, ES_bootstrap_mean], [0, max(f_bootstrap)], 'k-', 'LineWidth', 2);
    
    % Plot confidence interval lines
    plot([bootstrap_CI(1), bootstrap_CI(1)], [0, max(f_bootstrap)], 'm--', 'LineWidth', 2);
    plot([bootstrap_CI(2), bootstrap_CI(2)], [0, max(f_bootstrap)], 'm--', 'LineWidth', 2);
    
    % Annotations
    text(ES_bootstrap_mean, max(f_bootstrap)*0.9, sprintf(' Mean ES = %.2f', ES_bootstrap_mean), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
    text(bootstrap_CI(1), max(f_bootstrap)*0.9, sprintf(' Lower CI = %.2f', bootstrap_CI(1)), 'FontSize', 12, 'Color', 'm', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
    text(bootstrap_CI(2), max(f_bootstrap)*0.9, sprintf(' Upper CI = %.2f', bootstrap_CI(2)), 'FontSize', 12, 'Color', 'm', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    
    % Labels and Title
    xlabel('Expected Shortfall (ES)', 'FontSize', 14);
    ylabel('Density', 'FontSize', 14);
    title('Bootstrap ES Estimates', 'FontSize', 16);
    
    % Legend
    legend('Bootstrap ES Histogram', 'Kernel Density Estimate', 'Bootstrap Mean ES', 'Bootstrap CI', 'Location', 'Best');
    
    grid on;
    box on;
    
    % -------------------------------
    % 9. Final Adjustments
    % -------------------------------
    
    % Adjust figure layout
    sgtitle(sprintf('Expected Shortfall Analysis (df = %d, n = %d, B = %d, ESlevel = %.2f)', df, n, B, ESlevel), 'FontSize', 18);
    
    % -------------------------------
    % 10. Save the Figure (Optional)
    % -------------------------------
    
    % Uncomment the following line to save the figure
    % saveas(gcf, 'ES_Bootstrap_Analysis.png');
end

% Task4_01.m



% Function definition
function ES = compute_ES_student_t(nu, location, scale, ESlevel)
    % compute_ES_student_t calculates the Expected Shortfall (ES) for a Student's t-distribution.
    %
    % Syntax:
    %   ES = compute_ES_student_t(nu)
    %   ES = compute_ES_student_t(nu, location, scale, ESlevel)
    %
    % Inputs:
    %   nu       - Degrees of freedom (required)
    %   location - Location parameter (default: 0)
    %   scale    - Scale parameter (default: 1)
    %   ESlevel  - Tail probability (default: 0.05)
    %
    % Output:
    %   ES       - Expected Shortfall value
    %
    % Example:
    %   ES = compute_ES_student_t(5); % Uses default location, scale, and ESlevel
    %   ES = compute_ES_student_t(5, [], [], 0.01); % Uses default location and scale, ESlevel=0.01

    % Set default values if arguments are not provided or empty
    if ~exist('location', 'var') || isempty(location)
        location = 0;
    end
    if ~exist('scale', 'var') || isempty(scale)
        scale = 1;
    end
    if ~exist('ESlevel', 'var') || isempty(ESlevel)
        ESlevel = 0.05;
    end

    % Validate inputs
    if nu <= 0
        error('Degrees of freedom (nu) must be positive.');
    end
    if scale <= 0
        error('Scale parameter must be positive.');
    end
    if ESlevel <= 0 || ESlevel >= 1
        error('ESlevel must be between 0 and 1 (exclusive).');
    end

    % Compute the VaR for the standard t-distribution
    alpha = ESlevel;                % Tail probability
    VaR_standard = tinv(alpha, nu); % VaR for standard t-distribution (negative value)

    % Compute the PDF at VaR for the standard t-distribution
    pdf_VaR_standard = tpdf(VaR_standard, nu);

    % Compute ES for the standard t-distribution using the correct formula
    ES_standard = - ( ( nu + VaR_standard^2 ) / ( ( nu - 1 ) * alpha ) ) * pdf_VaR_standard;

    % Adjust ES for location and scale parameters
    ES = location + scale * ES_standard;

end

