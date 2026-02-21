
% Running the simulation for different degrees of freedom
total1 = zeros(1, 9); % For parametric coverage
total2 = zeros(1, 9); % For non-parametric coverage
totalL1 = zeros(1, 9); % For parametric CI lengths
totalL2 = zeros(1, 9); % For non-parametric CI lengths

for sigma = 2:15
    % Call the function and store the outputs in individual variables
    [actual90, actual91, length90, length91] = simulate_ES_bootstrap_performance(100, sigma, 25, 500, 'both');
    
    % Store the results in their respective arrays
    total1(sigma-1) = actual90;    % Parametric coverage
    total2(sigma-1) = actual91;    % Non-parametric coverage
    totalL1(sigma-1) = length90;   % Parametric CI length
    totalL2(sigma-1) = length91;   % Non-parametric CI length
end

% Plotting the coverage probabilities
figure;
plot(2:15, total1, 'LineWidth', 2);
hold on;
plot(2:15, total2, 'r', 'LineWidth', 2);
xlabel('Degrees of Freedom');
ylabel('Nominal Coverage Probabilities');
ylim([0.6, 1]);
legend('Parametric', 'Non-parametric');
hold off;

% Plotting the CI lengths
figure;
plot(2:15, totalL1, 'LineWidth', 2);
hold on;
plot(2:15, totalL2, 'r', 'LineWidth', 2);
xlabel('Degrees of Freedom');
ylabel('CI Length');
legend('Parametric', 'Non-parametric');
hold off;

function [actual90, actual91, length90, length91] = simulate_ES_bootstrap_performance(sim, df, n, B, method)
    if nargin < 1
        sim = 10; % default number of simulations
    end
    if nargin < 2
        df = 4; % default degrees of freedom
    end
    if nargin < 3
        n = 250; % default sample size
    end
    if nargin < 4
        B = 500; % default number of bootstrap replications
    end
    if nargin < 5
        method = 'both'; % default method
    end
    
    % Input validation
    if df <= 1
        error('Degrees of freedom (df) must be greater than 1.');
    end
    if n <= 0 || floor(n) ~= n
        error('Sample size n must be a positive integer.');
    end
    if B <= 0 || floor(B) ~= B
        error('Number of bootstrap replications B must be a positive integer.');
    end
    if sim <= 0 || floor(sim) ~= sim
        error('Number of simulations sim must be a positive integer.');
    end
    
    % Initialize Variables
    bool0 = zeros(1, sim);
    bool1 = zeros(1, sim);
    length_p = zeros(1, sim);
    length_np = zeros(1, sim);
    
    fprintf('Starting %d simulations...\n', sim);
    
    for s = 1:sim
        % Data Generation
        data = trnd(df, [n, 1]);
        % Compute True ES
        ES_true = compute_ES_student_t(df, 0, 1, 0.05);
        
        % Parametric Bootstrap CI
        if strcmp(method, 'parametric') || strcmp(method, 'both')
            try
                CI_parametric = compute_ES_parametric_bootstrap_single(data, B);
                fprintf('Parametric CI: [%.4f, %.4f]\n', CI_parametric(1), CI_parametric(2));
                bool0(s) = (ES_true > CI_parametric(1)) && (ES_true < CI_parametric(2));
                length_p(s) = abs(CI_parametric(2) - CI_parametric(1));
            catch
                warning('Parametric bootstrap failed.');
            end
        end
        
        % Non-Parametric Bootstrap CI
        if strcmp(method, 'non-parametric') || strcmp(method, 'both')
            try
                CI_nonparametric = compute_ES_nonparametric_bootstrap_single(data, B, n);
                fprintf('Non-parametric CI: [%.4f, %.4f]\n', CI_nonparametric(1), CI_nonparametric(2));
                bool1(s) = (ES_true > CI_nonparametric(1)) && (ES_true < CI_nonparametric(2));
                length_np(s) = abs(CI_nonparametric(2) - CI_nonparametric(1));
            catch
                warning('Non-parametric bootstrap failed.');
            end
        end
    end
    
    % Return the proportion of times ES_true is in the confidence interval and the lengths of the CIs
    actual90 = mean(bool0, 'omitnan');
    actual91 = mean(bool1, 'omitnan');
    length90 = mean(length_p, 'omitnan');
    length91 = mean(length_np, 'omitnan');
end


function CI = compute_ES_parametric_bootstrap_single(data, B)
    ESlevel = 0.05;
    pd = fitdist(data, 'tLocationScale');  % MLE of t-distribution
    nu_mle = pd.nu;
    location_mle = pd.mu;
    scale_mle = pd.sigma;
    
    ES_bootstrap_p = zeros(B, 1);
    
    for b = 1:B
        bootstrap_sample_p = random('tLocationScale', location_mle, scale_mle, nu_mle, length(data), 1);
        VaR_bootstrap_p = quantile(bootstrap_sample_p, ESlevel);
        ES_bootstrap_p(b) = mean(bootstrap_sample_p(bootstrap_sample_p <= VaR_bootstrap_p));
    end
    
    lower_p = quantile(ES_bootstrap_p, 0.05);
    upper_p = quantile(ES_bootstrap_p, 0.95);
    
    CI = [lower_p, upper_p];
end

function CI = compute_ES_nonparametric_bootstrap_single(data, B, n)
    ESlevel = 0.05;
    ES_bootstrap_np = zeros(B, 1);
    
    for b = 1:B
        bootsamp = datasample(data, n, 'Replace', true);
        VaR_bootstrap_np = quantile(bootsamp, ESlevel);
        ES_bootstrap_np(b) = mean(bootsamp(bootsamp <= VaR_bootstrap_np));
    end
    
    lower_np = quantile(ES_bootstrap_np, 0.05);
    upper_np = quantile(ES_bootstrap_np, 0.95);
    
    CI = [lower_np, upper_np];
end

function ES = compute_ES_student_t(nu, location, scale, ESlevel)
    if nu <= 0
        error('Degrees of freedom (nu) must be positive.');
    end
    if scale <= 0
        error('Scale parameter must be positive.');
    end
    if ESlevel <= 0 || ESlevel >= 1
        error('ESlevel must be between 0 and 1 (exclusive).');
    end
    
    alpha = ESlevel;
    VaR_standard = tinv(alpha, nu);
    pdf_VaR_standard = tpdf(VaR_standard, nu);
    
    ES_standard = -((nu + VaR_standard^2) / ((nu - 1) * alpha)) * pdf_VaR_standard;
    ES = location + scale * ES_standard;
end
