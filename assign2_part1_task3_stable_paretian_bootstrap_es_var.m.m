    

    % Example parameters
    df = 4;        % Degrees of freedom for Student's t-distribution
    n = 10000;     % Original data size
    B = 100;       % Number of bootstrap samples
    
    % Call the main function
    plot_ES_VaR_bootstrap(df, n, B);

function plot_ES_VaR_bootstrap(df, n, B)
        % plot_ES_VaR_bootstrap Computes and Plots VaR and ES Estimates via Bootstrap Methods
        %
        % Inputs:
        %   df - Degrees of freedom for Student's t-distribution
        %   n  - Original data size
        %   B  - Number of bootstrap samples
        %
        % Example:
        %   plot_ES_VaR_bootstrap(4, 10000, 100);
        
        %% -------------------------------
        % 1. Data Generation
        % -------------------------------
        
        % Parameters for the GAt distribution
        d = 1.71;
        v = 1.96;
        theta = 0.879;
        mu = 0.175;
        c = 1.21;
        
        % Number of observations and bootstrap samples
        % n = 10000;       % Original data size (passed as input)
        % B = 100;         % Number of bootstrap samples (passed as input)
        %gamma = 0.01;    % Confidence level for VaR and ES (1%)
        
        % Simulate data from the GAt distribution
        %z = GAtsim(n, d, v, theta);

      % 1. Data Generation
        % -------------------------------
        
        % Parameters for the stable Paretian distribution
        alpha = 1.7;  % Stability parameter (0 < alpha <= 2)
        beta = -0.6;   % Skewness parameter (-1 <= beta <= 1)
        scale = 1;    % Scale parameter (gamma > 0)
        delta = 0;    % Location parameter (real number)
        seed = 12345; % Random seed for reproducibility
        
        % Simulate data from the stable Paretian distribution
        data = stabgen(n, alpha, beta, scale, delta, seed);

        
        % -------------------------------
        % 2. Compute True ES
        % -------------------------------
         gamma = 0.01;  % Confidence level for VaR and ES (1%)
            
         [actual1, actual2] = asymstableES(gamma, alpha, beta, delta, scale, 1);
            % True VaR and ES cannot be analytically computed for stable Paretian
            % Distributions in general. Use empirical estimates or prior theoretical values.
            VaR_true = quantile(data, gamma);
            ES_true = mean(data(data <= VaR_true));

    fprintf('True Value at Risk (VaR): %.4f\n', actual2);
    fprintf('True Expected Shortfall (ES): %.4f\n', actual1);

            [param, stderr, iters, loglik, Varcov] = StableMLE(data);
                fprintf('True Expected Shortfall (ES): %.4f\n', param);
                
        %% -------------------------------
        % 3. Compute Bootstrap VaR and ES Values
        % -------------------------------
        
        % Parametric Bootstrap using Student's t-distribution
        [VaR_parametric_values, ES_parametric_values] = compute_parametric_bootstrap_single(data, B, n, gamma);
        
        % Non-Parametric Bootstrap
        [VaR_nonparametric_values, ES_nonparametric_values] = compute_nonparametric_bootstrap_single(data, B, n, gamma);
        
        % Bootstrap using Gaussian Mixture Model (Mixture of Normals)
        [VaR_mixture_values, ES_mixture_values] = mixture_of_normals(data, B, n, gamma);
        
        % Bootstrap using Gaussian (Normal Distribution)
        [VaR_gaussian_values, ES_gaussian_values] = Gaussian_bootstrap(data, B, n, gamma);
        
        [VaR_ES_GA_t, ES_GA_t] = compute_ES_parametric_bootstrap_GAt(data, B,n, gamma);


        fprintf('NCT estimation starts now...\n');
        [VaR_ES_NCT, ES_NCT] = compute_ES_parametric_bootstrap_NCT(data, B,n, gamma);

        %% -------------------------------
        % 4. Plot the Results (Box Plots)
        % -------------------------------
        
        % Prepare data for VaR Boxplot
        VaR_data = [VaR_nonparametric_values, VaR_gaussian_values, VaR_parametric_values, VaR_ES_GA_t, VaR_ES_NCT,VaR_mixture_values];
        VaR_labels = {'Non-Parametric', 'Gaussian', 'Student t','ESGAt','NCT', 'Mixture of Normals'};
        
        % Prepare data for ES Boxplot
        ES_data = [ES_nonparametric_values, ES_gaussian_values, ES_parametric_values, ES_GA_t,ES_NCT, ES_mixture_values];
        ES_labels = {'Non-Parametric', 'Gaussian', 'Student t','ESGAt','NCT' ,'Mixture of Normals'};
        
        % Create a figure with two subplots: one for VaR and one for ES
        figure('Name', 'VaR and ES Bootstrap Estimates', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 600]);
        
        % --- Subplot 1: VaR Boxplot ---
        subplot(1,2,1);
        boxplot(VaR_data, 'Labels', VaR_labels, 'Whisker', 1.5);
        hold on;
        
        % Add the horizontal line for the true VaR value
        yline(VaR_true, 'r-', 'LineWidth', 2, 'Label', 'True VaR', 'LabelHorizontalAlignment', 'left');
        
        % Customize the plot
        title('Bootstrap VaR Estimates Comparison');
        ylabel('Value at Risk (VaR)');
        xlabel('Bootstrap Method');
        grid on;
        hold off;
        
        % --- Subplot 2: ES Boxplot ---
        subplot(1,2,2);
        boxplot(ES_data, 'Labels', ES_labels, 'Whisker', 1.5);
        hold on;
        
        % Add the horizontal line for the true ES value
        yline(ES_true, 'r-', 'LineWidth', 2, 'Label', 'True ES', 'LabelHorizontalAlignment', 'left');
        
        % Customize the plot
        title('Bootstrap ES Estimates Comparison');
        ylabel('Expected Shortfall (ES)');
        xlabel('Bootstrap Method');
        grid on;
        hold off;
        
        % Enhance overall figure appearance
        sgtitle('Comparison of Bootstrap Methods for VaR and ES Estimates', 'FontSize', 16);
    end
    
    %% -------------------------------
    % Bootstrap Functions
    % -------------------------------
    
    function [VaR_values, ES_values] = compute_parametric_bootstrap_single(data, B, n, gamma)
        ESlevel = gamma;                % ES confidence level
        VaR_values = zeros(B, 1);       % Preallocate VaR estimates
        ES_values = zeros(B, 1);        % Preallocate ES estimates
    
        for b = 1:B
            ind = unidrnd(n, [n, 1]); 
            bootsamp = data(ind);      
    
            mle_params = mle(bootsamp, 'distribution', 'tLocationScale');
            nu_mle = mle_params(3);      % Degrees of freedom
            location_mle = mle_params(1);% Location parameter
            scale_mle = mle_params(2);   % Scale parameter
    
            % Compute ES as mean of losses beyond VaR
            %ES_values(b) = mean(bootstrap_sample_p(bootstrap_sample_p <= VaR_bootstrap_p));
            VaR_values(b) = location_mle + scale_mle * tinv(gamma, nu_mle);
            %ES_values(b) = location_mle - scale_mle * ((nu_mle + (tinv(gamma, nu_mle))^2) / (nu_mle - 1)) * tpdf(tinv(gamma, nu_mle), nu_mle) / gamma;    
            ES_values(b) =  mean(bootsamp(bootsamp <= VaR_values(b)));  
        end
    end
    
    function [VaR_values, ES_values] = compute_nonparametric_bootstrap_single(data, B, n, gamma)
        VaRlevel = gamma;               % VaR confidence level
        ESlevel = gamma;                % ES confidence level
        VaR_values = zeros(B, 1);       % Preallocate VaR estimates
        ES_values = zeros(B, 1);        % Preallocate ES estimates
    
        for b = 1:B
            % Generate bootstrap sample indices
            ind = unidrnd(n, [n, 1]);  % Bootstrap sample indices
            bootsamp = data(ind);        % Bootstrap sample
    
            % Compute VaR for bootstrap sample
            VaR_bootstrap_np = quantile(bootsamp, VaRlevel);
            VaR_values(b) = VaR_bootstrap_np;
    
            % Compute ES as mean of losses beyond VaR
            ES_values(b) = mean(bootsamp(bootsamp <= VaR_bootstrap_np));
        end
    end
    
    function [VaR_values, ES_values] = mixture_of_normals(data, B, n, gamma)
        % mixture_of_normals Computes VaR and ES via Gaussian Mixture Model Bootstrap
        %
        % Inputs:
        %   data - Original dataset (n x 1 vector)
        %   B    - Number of bootstrap samples
        %   n    - Number of observations in each bootstrap sample
        %   gamma - Confidence level (e.g., 0.01 for 1%)
        %
        % Outputs:
        %   VaR_values - Estimated VaR values from the mixture of normals bootstrap
        %   ES_values  - Estimated ES values from the mixture of normals bootstrap
    
        ESlevel = gamma;                % ES confidence level
        VaR_values = zeros(B, 1);       % Preallocate VaR estimates
        ES_values = zeros(B, 1);        % Preallocate ES estimates
    
        for b = 1:B
            % Generate bootstrap sample indices
            ind = unidrnd(n, [n, 1]);  % Bootstrap sample indices
            bootsamp = data(ind);        % Bootstrap sample
            bootsamp = bootsamp(:);      % Ensure column vector
    
            try
                % Fit Gaussian Mixture Model with 2 components
                gm = fitgmdist(bootsamp, 2, 'Replicates', 5, 'Options', statset('MaxIter', 1000));
    
                % Generate samples from the fitted mixture model
                sample_mixnorm = random(gm, n);
    
                % Compute VaR for bootstrap sample
                VaR_bootstrap_p = quantile(sample_mixnorm, ESlevel);
                VaR_values(b) = VaR_bootstrap_p;
    
                % Compute ES as mean of losses beyond VaR
                ES_values(b) = mean(sample_mixnorm(sample_mixnorm <= VaR_bootstrap_p));
            catch ME
                % If fitting fails, assign NaN and optionally display a warning
                VaR_values(b) = NaN;
                ES_values(b) = NaN;
                warning('GMM fitting failed on bootstrap sample %d: %s', b, ME.message);
            end
        end
    
        % Remove any NaN values resulting from failed fits
        valid_indices = ~isnan(VaR_values) & ~isnan(ES_values);
        VaR_values = VaR_values(valid_indices);
        ES_values = ES_values(valid_indices);
    end
    
    function [VaR_values, ES_values] = Gaussian_bootstrap(data, B, n, gamma)
        % Gaussian_bootstrap Computes VaR and ES via Gaussian (Normal) Bootstrap
        %
        % Inputs:
        %   data - Original dataset (n x 1 vector)
        %   B    - Number of bootstrap samples
        %   n    - Number of observations in each bootstrap sample
        %   gamma - Confidence level (e.g., 0.01 for 1%)
        %
        % Outputs:
        %   VaR_values - Estimated VaR values from the Gaussian bootstrap
        %   ES_values  - Estimated ES values from the Gaussian bootstrap
    
        ESlevel = gamma;                % ES confidence level
        VaR_values = zeros(B, 1);       % Preallocate VaR estimates
        ES_values = zeros(B, 1);        % Preallocate ES estimates
    
        for b = 1:B
            % Generate bootstrap sample indices
            ind = unidrnd(n, [n, 1]);  % Bootstrap sample indices
            bootsamp = data(ind);        % Bootstrap sample
    
            % Fit Gaussian (Normal) distribution
            [mu_hat, sigma_hat] = normfit(bootsamp);
            
            % Compute VaR for bootstrap sample
            VaR_normal = norminv(ESlevel, mu_hat, sigma_hat);
            VaR_values(b) = VaR_normal;
    
            % Compute ES as per normal distribution formula
            ES_normal = mu_hat - sigma_hat * normpdf(norminv(gamma)) / gamma;
            ES_values(b) = ES_normal;
        end
    end

    function [VaR_values, ES_values] = compute_ES_parametric_bootstrap_GAt(data, B,n, gamma)

    % Initialize array to store ES estimates
        ESlevel = gamma;                % ES confidence level
        VaR_values = zeros(B, 1);       % Preallocate VaR estimates
        ES_values = zeros(B, 1);        % Preallocate ES estimates

    for b = 1:B
        % Fit GAt parameters using MLE
        ind = unidrnd(n, [n, 1]);  % Bootstrap sample indices
        bootsamp = data(ind);        % Bootstrap sample

        [param_est, ~, ~, ~, ~] = GAtestimation(bootsamp, 0.01, [], []);
        d_est = param_est(1);
        v_est = param_est(2);
        theta_est = param_est(3);
        mu_est = param_est(4);
        c_est = param_est(5);

        VaR_values(b) = mu_est + c_est * GAtquantile(gamma, d_est, v_est, theta_est);
        %ES_values(b) = mean(bootsamp(bootsamp <= VaR_values(b)));

        %L = v_est / (v_est + (-VaR_values(b) * theta_est)^d_est);
        
        VaR_non_location = GAtquantile(gamma, d_est, v_est, theta_est);
        L = v_est / (v_est + (-VaR_non_location * theta_est)^d_est);
        % Step 3: Define shape parameters for numerator and denominator
        a_num = v_est - (1/d_est);
        b_num = 2/d_est;
        a_den = v_est;
        b_den = 1/d_est;
                
        % Step 4: Compute the incomplete beta functions
        % B_L_num = B_L(v - 1/d, 2/d)
        B_L_num = betainc(L, a_num, b_num) * beta(a_num, b_num);
        
        % B_L_den = B_L(v, 1/d)
        B_L_den = betainc(L, a_den, b_den) * beta(a_den, b_den);
        

        ES_values(b) = mu_est + c_est * ((((-1)*v_est^(1/d_est))*(1+theta_est^2)) / ((1 + theta_est^2) * theta_est)) * (B_L_num / B_L_den);
        % Compute ES for bootstrap sample

    end
    
    % Remove any NaN values resulting from failed fits or computations
    ES_values = ES_values(~isnan(ES_values));
    VaR_values = VaR_values(~isnan(VaR_values));
    end
    
function [VaR_values, ES_values] = compute_ES_parametric_bootstrap_NCT(data, B, n, gamma)

    % Initialize arrays to store estimates
    VaR_values = zeros(B, 1);
    ES_values = zeros(B, 1);

    for b = 1:B
        fprintf('Starting NCT estimation for bootstrap sample %d out of %d...\n', b, B);
        % Generate bootstrap sample
        ind = unidrnd(n, [n, 1]);
        bootsamp = data(ind);

        % Estimate NCT parameters using MLE
        [param_est, ~, ~, ~, ~] = NCTestimation(bootsamp);
        nu_hat = param_est(1);
        delta_hat = param_est(2);
        mu_hat = param_est(3);
        sigma_hat = param_est(4);

        % Compute VaR
        VaR_standard = nctinv(gamma, nu_hat, delta_hat);
        VaR_values(b) = mu_hat + sigma_hat * VaR_standard;

        % Compute ES using numerical integration
        ES_values(b) = compute_ES_NCT(nu_hat, delta_hat, mu_hat, sigma_hat, gamma);

        % Handle potential NaN values
        if isnan(ES_values(b)) || isnan(VaR_values(b))
            ES_values(b) = mean(bootsamp(bootsamp <= VaR_values(b)));
        end
    end

    % Remove any NaN values resulting from failed fits or computations
    valid_indices = ~isnan(VaR_values) & ~isnan(ES_values);
    VaR_values = VaR_values(valid_indices);
    ES_values = ES_values(valid_indices);
end


    
    %% -------------------------------
    % Supporting Functions
    % -------------------------------
    
    function ES = compute_ES_student_t(nu, location, scale, ESlevel)
        % compute_ES_student_t Computes Expected Shortfall for Student's t-distribution
        %
        % Inputs:
        %   nu       - Degrees of freedom
        %   location - Location parameter (default: 0)
        %   scale    - Scale parameter (default: 1)
        %   ESlevel  - Confidence level (e.g., 0.01 for 1%)
        %
        % Output:
        %   ES - Expected Shortfall
    
        % Set default values if arguments are not provided or empty
        if ~exist('location', 'var') || isempty(location)
            location = 0;
        end
        if ~exist('scale', 'var') || isempty(scale)
            scale = 1;
        end
        if ~exist('ESlevel', 'var') || isempty(ESlevel)
            ESlevel = 0.01;
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
        alpha = ESlevel;  % Tail probability
        VaR_standard = tinv(alpha, nu);  % VaR for standard t-distribution (negative value)
    
        % Compute the PDF at VaR for the standard t-distribution
        pdf_VaR_standard = tpdf(VaR_standard, nu);
    
        % Compute ES for the standard t-distribution using the correct formula
        ES_standard = -((nu + VaR_standard^2) / ((nu - 1) * alpha)) * pdf_VaR_standard;
    
        % Adjust ES for location and scale parameters
        ES = location + scale * ES_standard;
    end
    
    function VaR = compute_VaR_student_t(nu, location, scale, VaRlevel)
        % compute_VaR_student_t Computes Value at Risk for Student's t-distribution
        %
        % Inputs:
        %   nu       - Degrees of freedom
        %   location - Location parameter (default: 0)
        %   scale    - Scale parameter (default: 1)
        %   VaRlevel - Confidence level (e.g., 0.01 for 1%)
        %
        % Output:
        %   VaR - Value at Risk
    
        % Set default values if arguments are not provided or empty
        if ~exist('location', 'var') || isempty(location)
            location = 0;
        end
        if ~exist('scale', 'var') || isempty(scale)
            scale = 1;
        end
        if ~exist('VaRlevel', 'var') || isempty(VaRlevel)
            VaRlevel = 0.01;
        end
    
        % Validate inputs
        if nu <= 0
            error('Degrees of freedom (nu) must be positive.');
        end
        if scale <= 0
            error('Scale parameter must be positive.');
        end
        if VaRlevel <= 0 || VaRlevel >= 1
            error('VaRlevel must be between 0 and 1 (exclusive).');
        end
    
        % Compute VaR for Student's t-distribution
        VaR = tinv(VaRlevel, nu) * scale + location;
    end
    
    function x = GAtsim(sim, d, v, theta)

        x = zeros(sim,1);
        lo = 1e-6; hi = 1 - lo;
        for i = 1:sim
            u = rand;
            u = max(u, lo);
            u = min(u, hi);
            x(i) = GAtquantile(u, d, v, theta);
        end
    end
    
    function q = GAtquantile(p, d, v, theta)
        % GAtquantile Computes the quantile function of the GAt distribution
        %
        % Inputs:
        %   p     - Probability level
        %   d     - Parameter d
        %   v     - Parameter v
        %   theta - Parameter theta
        %
        % Output:
        %   q - Quantile corresponding to probability p
    
        % Initialize search bounds
        lobound = -100;
        hibound = 100;
    
        % Adjust lobound until ff(lobound) <= 0
        while true
            [~, cdf_lobound] = GAt(lobound, d, v, theta);
            f_lobound = cdf_lobound - p;
            if f_lobound <= 0
                break;
            else
                lobound = lobound - 10;
                if lobound < -1e6
                    error('Cannot find suitable lower bound for fzero.');
                end
            end
        end
    
        % Adjust hibound until ff(hibound) >= 0
        while true
            [~, cdf_hibound] = GAt(hibound, d, v, theta);
            f_hibound = cdf_hibound - p;
            if f_hibound >= 0
                break;
            else
                hibound = hibound + 10;
                if hibound > 1e6
                    error('Cannot find suitable upper bound for fzero.');
                end
            end
        end
    
        % Use fzero to solve for the quantile
        tol = 1e-8;
        opt = optimset('display', 'off', 'TolX', tol);
        q = fzero(@(x) ff(x, p, d, v, theta), [lobound, hibound], opt);
    
        function out = ff(x, p, d, v, theta)
            [~, cdf_val] = GAt(x, d, v, theta);
            out = cdf_val - p;
        end
    end
    
    function [pdf, cdf, themean, thevar, quant, theES]= ...
         GAt(xvec,d,v,theta, gammaforES)

ll=length(xvec); xvec=reshape(xvec,ll,1); pdf = zeros(ll,1);
konst = 1 / ( beta(1/d,v) * v^(1/d) * (theta+1/theta) / d );
k = find(xvec<0);
if any(k), y=xvec(k); pdf(k) = ( 1 + (-y*theta).^d / v ).^(-(v+1/d)); end
k = find(xvec>=0);
if any(k), y=xvec(k); pdf(k) = ( 1 + (y/theta).^d / v ).^(-(v+1/d)); end
pdf = konst * pdf;
if nargout>1
  cdf=zeros(length(xvec),1);
  k = find(xvec<0);
  if any(k)
    y=xvec(k); L = v./(v+(-y*theta).^d);
    cdf(k) = betainc(L,v,1/d)/(1+theta^2);
  end
  k = find(xvec==0);
  if any(k), y=xvec(k); cdf(k) = 1/(1+theta^2); end
  k = find(xvec>0);
  if any(k)
    y=xvec(k); top=(y/theta).^d; U=top./(v+top);
    cdf(k) = 1/(1+theta^2) + betainc(U,1/d,v)/(1+theta^(-2));
  end
end


if nargout>2
   themean = NaN; thevar=NaN; theES=NaN;
   if v*d>1
     themean=GAtmom(1,d,v,theta); 
     quant = GAtquantile(gammaforES, d,v,theta);
     theES = GAttail(1,d,v,theta, quant); 
   end
   if v*d>2, thevar=GAtmom(2,d,v,theta) - themean^2; end
end

function m = GAtmom(r,d,v,theta)
t1 = ( (-1)^r * theta^(-r-1) + theta^(r+1) ) / (theta + 1/theta);
t2 = beta((r+1)/d , v-r/d) / beta(1/d,v) * v^(r/d);
m = t1 * t2;
end

function m = GAttail(r,d,v,theta, c)
t1 = (-1)^r * v^(r/d) * (1+theta^2) / (theta^r + theta^(r+2));
L = v / ( v+(-c*theta)^d );
t2 = beta(v-r/d, (r+1)/d) * betainc(L, v-r/d, ...
     (r+1)/d) / betainc(L,v,1/d) / beta(v,1/d);
m = t1*t2;
end

end

    
   function ES = compute_ES_GA_t(d, v, theta, gamma)
    % compute_ES_GA_t Computes Expected Shortfall for GAt distribution
    %
    % Inputs:
    %   d     - Parameter d of the GAt distribution
    %   v     - Parameter v of the GAt distribution
    %   theta - Parameter theta of the GAt distribution
    %   gamma - Confidence level for ES (e.g., 0.01 for 1%)
    %
    % Output:
    %   ES    - Expected Shortfall
    
    % Ensure gamma is within (0,1)
    if gamma <= 0 || gamma >= 1
        error('Confidence level gamma must be between 0 and 1 (exclusive).');
    end
    
    % Step 1: Compute VaR at level gamma using the GAt quantile function
    VaR = GAtquantile(gamma, d, v, theta);
    
    % Step 2: Compute L = v / (v + (-VaR * theta)^d)
    L = v / (v + (-VaR * theta)^d);
    
    % Step 3: Define shape parameters for numerator and denominator
    a_num = v - (1/d);
    b_num = 2/d;
    a_den = v;
    b_den = 1/d;
    
    % Validate that the shape parameters are positive
    if a_num <= 0
        error('Parameter a_num = v - 1/d must be positive.');
    end
    
    % Step 4: Compute the incomplete beta functions
    % B_L_num = B_L(v - 1/d, 2/d)
    B_L_num = betainc(L, a_num, b_num) * beta(a_num, b_num);
    
    % B_L_den = B_L(v, 1/d)
    B_L_den = betainc(L, a_den, b_den) * beta(a_den, b_den);
    
    % Step 5: Compute the ES using the analytical formula
    ES = ((((-1)*v^(1/d))*(1+theta^2)) / ((1 + theta^2) * theta)) * (B_L_num / B_L_den);
    
   end

  
function [param,stderr,iters,loglik,Varcov] = GAtestimation(x,vlo,initvec,fixd)
% [param,stderr,iters,loglik,Varcov] = GAtestimation(x,vlo,initvec,fixd)
% pass fixd as the value to fix the parameter d,
%   default is [], which indicates to estimate it

if nargin<2, vlo=[]; end
if nargin<3, initvec=[]; end
if nargin<4, fixd=[]; end

if isempty(vlo), vlo=0.01; end

if isempty(initvec)
  versuch=3; vhi=4; loglik=-Inf;
  vvec=linspace(vlo+0.02,vhi,versuch);
  for i=1:versuch
    vv=vvec(i);
    if isempty(fixd), initvec = [2 vv 0.98 0 3]; else initvec = [vv 0.98 0 3]; end
    [param0,stderr0,iters0,loglik0,Varcov0] = GAtestimation(x,vlo,initvec,fixd);
    if loglik0>loglik
      loglik=loglik0;
      param=param0; stderr=stderr0; iters=iters0; Varcov=Varcov0;
    end
  end
  return
end
if isempty(fixd)
  %%%%%%%%       d   v  theta  mu    c
  bound.lo=   [0.1 vlo    0.2  -1  1e-4];
  bound.hi=   [ 30 100      3   2  1e+4];
  bound.which=[  1   1      1   0     1];
else
  %%%%%%%%       v  theta  mu     c
  bound.lo=   [vlo    0.2  -1  1e-4];
  bound.hi=   [100      3   2  1e+4];
  bound.which=[  1      1   0     1];
end
nobs=length(x);
maxiter=length(initvec)*100; tol=1e-8; MaxFunEvals=length(initvec)*400;
opts=optimset('Display','none','Maxiter',maxiter,'TolFun',tol,'TolX',tol,'MaxFunEvals',MaxFunEvals,'LargeScale','Off');
if 1==1
  [pout,fval,exitflag,theoutput,grad,hess]= ...
    fminunc(@(param) GAtloglik(param,x,fixd,bound),einschrk(initvec,bound),opts);
else
  [pout,fval,exitflag,theoutput]= ...
    fminsearch(@(param) GAtloglik(param,x,fixd,bound),einschrk(initvec,bound),opts);
  hess=eye(length(pout));
end

V=inv(hess)/nobs; % Don't negate because we work with the negative of the log lik
[param,V]=einschrk(pout,bound,V);  % transform back and apply delta method to get V
param=param'; Varcov=V;
stderr=sqrt(diag(V));  % The resulting approximate standard errors of the parameters
loglik=-fval*nobs;          % The value of the log likelihood function at its maximum.
iters=theoutput.iterations;  % Required number of log likelihood function evaluations

function ll=GAtloglik(param,x,fixd,bound)
if nargin<4, bound=0; end
if isstruct(bound), paramvec=einschrk(real(param),bound,999); else paramvec=param; end
if isempty(fixd)
  d=paramvec(1); v=paramvec(2); theta=paramvec(3); mu=paramvec(4); c=paramvec(5);
else
  d=fixd; v=paramvec(1); theta=paramvec(2); mu=paramvec(3); c=paramvec(4);
end
z=(x-mu)/c; pdf = GAt(z,d,v,theta) / c;
llvec=log(pdf); ll=-mean(llvec);
if isinf(ll), ll=1e5; end

end
end

 function [y, V_out] = einschrk(x, bound, V_in)
    % einschrk Applies parameter constraints based on bounds
    %
    % Inputs:
    %   x      - Parameter vector
    %   bound  - Structure with fields:
    %            bound.lo  - Lower bounds
    %            bound.hi  - Upper bounds
    %            bound.which - Logical array indicating which parameters to fix
    %   V_in   - Variance-covariance matrix (optional)
    %
    % Outputs:
    %   y      - Constrained parameter vector
    %   V_out  - Adjusted variance-covariance matrix (if V_in is provided)
    
    y = x;
    for i = 1:length(x)
        if y(i) < bound.lo(i)
            y(i) = bound.lo(i);
        elseif y(i) > bound.hi(i)
            y(i) = bound.hi(i);
        end
    end
    
    if nargin == 3 && ~isempty(V_in)
        V_out = V_in;
        for i = 1:length(y)
            if bound.which(i) == 0
                V_out(i, :) = 0;
                V_out(:, i) = 0;
            end
        end
    else
        V_out = [];
    end
 end 

 function [nu_hat, delta_hat, mu_hat, sigma_hat] = NCT_quantile_estimation(data, nct_lookup_table, nu_grid, delta_grid)

    if nargin < 2
        [nct_lookup_table, nu_grid, delta_grid] = precompute_NCT_quantile_table();
    end

    % Standardize data
    mu_hat = mean(data);
    sigma_hat = std(data);
    data_std = (data - mu_hat) / sigma_hat;

    % Compute sample quantiles
    quantile_probs = [0.05, 0.25, 0.5, 0.75, 0.95];
    sample_quantiles = quantile(data_std, quantile_probs);

    % Initialize error matrix
    error_matrix = zeros(length(nu_grid), length(delta_grid));

    % Loop over grid of nu and delta
    for i = 1:length(nu_grid)
        for j = 1:length(delta_grid)
            theoretical_quantiles = squeeze(nct_lookup_table(i, j, :));
            error = sum((sample_quantiles - theoretical_quantiles').^2);
            error_matrix(i, j) = error;
        end
    end

    % Find the nu and delta that minimize the error
    [~, idx] = min(error_matrix(:));
    [i_min, j_min] = ind2sub(size(error_matrix), idx);
    nu_hat = nu_grid(i_min);
    delta_hat = delta_grid(j_min);
end

function [nct_lookup_table, nu_grid, delta_grid] = precompute_NCT_quantile_table()
    % precompute_NCT_quantile_table Precomputes a table of NCT quantiles
    %
    % Outputs:
    %   nct_lookup_table - 3D array of quantiles for each (nu, delta) pair
    %   nu_grid          - Grid of nu values used in the table
    %   delta_grid       - Grid of delta values used in the table

    % Define grids for nu and delta
    nu_grid = linspace(1, 30, 30);          % Degrees of freedom from 1 to 30
    delta_grid = linspace(-5, 5, 50);       % Noncentrality parameter from -5 to 5

    quantile_probs = [0.05, 0.25, 0.5, 0.75, 0.95]; % Quantiles to compute

    % Preallocate the lookup table
    nct_lookup_table = zeros(length(nu_grid), length(delta_grid), length(quantile_probs));

    % Compute theoretical quantiles for each (nu, delta) pair
    for i = 1:length(nu_grid)
        nu = nu_grid(i);
        for j = 1:length(delta_grid)
            delta = delta_grid(j);
            % Compute quantiles
            quantiles = nctinv(quantile_probs, nu, delta);
            nct_lookup_table(i, j, :) = quantiles;
        end
    end
end

function [param, stderr, iters, loglik, Varcov] = NCTestimation(x, initvec)
    % NCTestimation Estimates the parameters of the NCT distribution via MLE
    %
    % Inputs:
    %   x        - Data vector
    %   initvec  - Initial parameter estimates [nu, delta, mu, sigma]
    %
    % Outputs:
    %   param    - Estimated parameters [nu, delta, mu, sigma]
    %   stderr   - Standard errors of the estimates
    %   iters    - Number of iterations
    %   loglik   - Log-likelihood at the optimum
    %   Varcov   - Variance-covariance matrix of the estimates

    % Set default initial values if not provided
    if nargin < 2 || isempty(initvec)
        nu_init = 5;              % Degrees of freedom
        delta_init = 0;           % Noncentrality parameter
        mu_init = mean(x);        % Location parameter
        sigma_init = std(x);      % Scale parameter
        initvec = [nu_init, delta_init, mu_init, sigma_init];
    end

    % Parameter bounds
    % We can define bounds and use 'einschrk' function to enforce them
    % For parameters:
    %   nu > 0
    %   sigma > 0
    % No constraints on delta and mu

    bound.lo = [1e-3, -Inf, -Inf, 1e-3];
    bound.hi = [100, Inf, Inf, Inf];
    bound.which = [1, 0, 0, 1]; % Indicate which parameters are constrained

    % Optimization options
    nobs = length(x);
    maxiter = length(initvec) * 100;
    tol = 1e-8;
    MaxFunEvals = length(initvec) * 400;
    opts = optimset('Display', 'none', 'MaxIter', maxiter, 'TolFun', tol, 'TolX', tol, 'MaxFunEvals', MaxFunEvals, 'LargeScale', 'Off');

    % Use fminunc as in GAtestimation
    [pout, fval, exitflag, output, grad, hess] = fminunc(@(param) NCTnegloglik(param, x, bound), einschrk(initvec, bound), opts);

    % Check if hess is square
    if size(hess,1) ~= size(hess,2)
        warning('Hessian is not square. Cannot compute standard errors.');
        Varcov = NaN;
        stderr = NaN(size(pout));
    else
        % Compute variance-covariance matrix
        Varcov = inv(hess) / nobs;
        % Transform parameters back and apply delta method to get Varcov
        [param, Varcov] = einschrk(pout, bound, Varcov);
        stderr = sqrt(diag(Varcov));
    end

    param = param';
    loglik = -fval * nobs;
    iters = output.iterations;
end

function nll = NCTnegloglik(param, x, bound)
    % Negative log-likelihood function for the NCT distribution

    if isstruct(bound)
        paramvec = einschrk(real(param), bound, 999);
    else
        paramvec = param;
    end

    nu = paramvec(1);
    delta = paramvec(2);
    mu = paramvec(3);
    sigma = paramvec(4);

    % Constraints: nu > 0, sigma > 0
    if nu <= 0 || sigma <= 0
        nll = 1e5;
        return;
    end

    % Standardize data
    z = (x - mu) / sigma;

    % Compute log-likelihood
    ll = log(nctpdf(z, nu, delta)) - log(sigma);

    % Handle non-finite values
    if any(~isfinite(ll))
        nll = 1e5;
    else
        nll = -mean(ll);
    end
end

function y = nctpdf_fast(x, nu, delta)
% nctpdf_fast Efficient computation of NCT PDF
% This function computes the PDF of the noncentral t-distribution using
% an approximation or an efficient series expansion.

% For moderate values of delta, the built-in function is efficient enough.
% If delta is large, consider using an approximation.

y = nctpdf(x, nu, delta);

% Alternatively, implement a custom approximation if necessary.

end

function ES = compute_ES_NCT(nu, delta, mu, sigma, gamma)
    % compute_ES_NCT Computes Expected Shortfall for NCT distribution
    %
    % Inputs:
    %   nu    - Degrees of freedom
    %   delta - Noncentrality parameter
    %   mu    - Location parameter
    %   sigma - Scale parameter
    %   gamma - Confidence level
    %
    % Output:
    %   ES - Expected Shortfall

    % Use numerical integration to compute ES
    VaR = mu + sigma * nctinv(gamma, nu, delta);

    % Define the integrand
    integrand = @(x) x .* nctpdf((x - mu) / sigma, nu, delta) / sigma;

    % Compute ES via numerical integration from -Inf to VaR
    ES = (1 / gamma) * integral(integrand, -Inf, VaR);

    % Handle cases where integral fails
    if isnan(ES) || isinf(ES)
        % Use a sample simulation as a fallback
        N = 1e5;
        samples = mu + sigma * nctrnd(nu, delta, N, 1);
        ES = mean(samples(samples <= VaR));
    end
end

function x = stabgen(nobs, a, b, c, d, seed)
    % stabgen Generates random samples from an asymmetric stable Paretian distribution
    % Inputs:
    %   nobs - Number of observations
    %   a    - Stability parameter (0 < a <= 2)
    %   b    - Skewness parameter (-1 <= b <= 1)
    %   c    - Scale parameter (c > 0)
    %   d    - Location parameter (real number)
    %   seed - Random seed for reproducibility
    % Output:
    %   x    - Random samples from the stable Paretian distribution

    if nargin < 3, b = 0; end
    if nargin < 4, c = 1; end
    if nargin < 5, d = 0; end
    if nargin < 6, seed = rand; end

    z = nobs;
    rand('twister', seed);
    V = unifrnd(-pi/2, pi/2, 1, z);
    rand('twister', seed + 42);
    W = exprnd(1, 1, z);

    if a == 1
        x = (2 / pi) * (((pi / 2) + b * V) .* tan(V) - b * log((W .* cos(V)) ./ ((pi / 2) + b * V)));
        x = c * x + d - (2 / pi) * d * log(d) * c * b;
    else
        Cab = atan(b * tan(pi * a / 2)) / a;
        Sab = (1 + b^2 * (tan((pi * a) / 2))^2)^(1 / (2 * a));
        A = (sin(a * (V + Cab))) ./ ((cos(V)).^(1 / a));
        B0 = (cos(V - a * (V + Cab))) ./ W;
        B = abs(B0).^((1 - a) / a);
        x = Sab * A .* (B .* sign(B0));
        x = c * x + d;
    end
end

function [ES, VaR] = asymstableES(xi, a, b, mu, scale, method)
    % asymstableES Computes Expected Shortfall (ES) and Value at Risk (VaR)
    % for an asymmetric stable Paretian distribution.
    %
    % Inputs:
    %   xi    - Confidence level (e.g., 0.01 for 1%)
    %   a     - Stability parameter (0 < a <= 2)
    %   b     - Skewness parameter (-1 <= b <= 1)
    %   mu    - Location parameter
    %   scale - Scale parameter
    %   method - Method to compute ES (1: Stoyanov, 2: Tail Integration)
    %
    % Outputs:
    %   ES    - Expected Shortfall
    %   VaR   - Value at Risk

    if nargin < 3, b = 0; end
    if nargin < 4, mu = 0; end
    if nargin < 5, scale = 1; end
    if nargin < 6, method = 1; end

    % Find the quantile `q` of the stable Paretian distribution
    opt = optimset('Display', 'off', 'TolX', 1e-6);
    q = fzero(@(x) stabcdfroot(x, xi, a, b), -6, opt);
    VaR = mu + scale * q;

    % Compute Expected Shortfall (ES)
    if q == 0
        % Analytical formula for ES when q == 0
        t0 = (1 / a) * atan(b * tan(pi * a / 2));
        ES = ((2 * gamma((a - 1) / a)) / (pi - 2 * t0)) * (cos(t0) / cos(a * t0)^(1 / a));
        return;
    end

    if method == 1
        % Stoyanov method for computing ES
        ES = (scale * Stoy(q, a, b) / xi) + mu;
    else
        % Tail integration method
        ES = (scale * stabletailcomp(q, a, b) / xi) + mu;
    end
end

function diff = stabcdfroot(x, xi, a, b)
    % stabcdfroot Computes the difference between CDF and confidence level
    % for root-finding to determine the quantile.
    if exist('stableqkcdf', 'file')
        F = stableqkcdf(x, [a, b], 1); % Nolan's routine if available
    else
        [~, F] = asymstab(x, a, b);    % Alternative routine
    end
    diff = F - xi;
end

function tailcomp = stabletailcomp(q, a, b)
    % stabletailcomp Computes the tail integral for ES
    K = (a / pi) * sin(pi * a / 2) * gamma(a) * (1 - b); % K constant
    ell = -120; % Lower integration limit for numerical stability
    M = ell;    % Arbitrary large negative value
    term1 = K * (-M)^(1 - a) / (1 - a);
    term3 = quadl(@(x) stableCVARint(x, a, b), ell, q, 1e-5);
    tailcomp = term1 + term3;
end

function g = stableCVARint(x, a, b)
    % stableCVARint Computes the integrand for tail ES calculation
    if exist('stableqkpdf', 'file')
        den = stableqkpdf(x, [a, b], 1); % Nolan's routine if available
    else
        den = asymstab(x, a, b)';        % Alternative routine
    end
    g = x .* den;
end

function S = Stoy(cut, alpha, beta)
    % Stoy Computes the Stoyanov integral for ES
    cut = -cut; % Sign convention adjustment
    bbar = -sign(cut) * beta;
    t0bar = (1 / alpha) * atan(bbar * tan(pi * alpha / 2));
    small = 1e-8; tol = 1e-8; abscut = abs(cut);
    integ = quadl(@(t) stoyint(t, abscut, alpha, t0bar), -t0bar + small, pi / 2 - small, tol);
    S = alpha / (alpha - 1) / pi * abscut * integ;
end

function I = stoyint(t, cut, a, t0bar)
    % stoyint Computes the Stoyanov integrand
    s = t0bar + t;
    g = sin(a * s - 2 * t) ./ sin(a * s) - a * cos(t).^2 ./ sin(a * s).^2;
    v = (cos(a * t0bar)).^(1 / (a - 1)) .* (cos(t) ./ sin(a * s)).^(a / (a - 1)) ...
        .* cos(a * s - t) ./ cos(t);
    term = -(abs(cut)^(a / (a - 1)));
    I = g .* exp(term .* v);
end

function [f, F] = asymstab(xvec, a, b)
    % asymstab Computes the PDF and optionally the CDF of the asymmetric stable Paretian
    border_tol = 1e-8; lo = border_tol; hi = 1 - border_tol; tol = 1e-7;
    xl = length(xvec); F = zeros(xl, 1); f = F;
    for loop = 1:length(xvec)
        x = xvec(loop);
        f(loop) = quadl(@(u) fff(u, x, a, b, 1), lo, hi, tol) / pi;
        if nargout > 1
            F(loop) = 0.5 - (1 / pi) * quadl(@(u) fff(u, x, a, b, 0), lo, hi, tol);
        end
    end
end

function I = fff(uvec, x, a, b, dopdf)
    % fff Helper function for integration in asymstab
    subs = 1; I = zeros(size(uvec));
    for ii = 1:length(uvec)
        u = uvec(ii);
        t = (1 - u) / u;
        if a == 1
            cf = exp(-abs(t) * (1 + 1i * b * (2 / pi) * sign(t) * log(t)));
        else
            cf = exp(-(abs(t)^a) * (1 - 1i * b * sign(t) * tan(pi * a / 2)));
        end
        z = exp(-1i * t * x) .* cf;
        if dopdf == 1
            g = real(z);
        else
            g = imag(z) ./ t;
        end
        I(ii) = g * u^(-2);
    end
end



function [param, stderr, iters, loglik, Varcov] = StableMLE(x, initvec)
    % StableMLE Estimates the parameters of the stable Paretian distribution via MLE.
    %
    % Inputs:
    %   x        - Data vector
    %   initvec  - Initial parameter estimates [alpha, beta, mu, scale]
    %
    % Outputs:
    %   param    - Estimated parameters [alpha, beta, mu, scale]
    %   stderr   - Standard errors of the estimates
    %   iters    - Number of iterations
    %   loglik   - Log-likelihood at the optimum
    %   Varcov   - Variance-covariance matrix of the estimates

    % Set default initial values if not provided
    if nargin < 2 || isempty(initvec)
        alpha_init = 1.7;         % Stability parameter
        beta_init = 0;            % Skewness parameter
        mu_init = mean(x);        % Location parameter
        scale_init = std(x);      % Scale parameter
        initvec = [alpha_init, beta_init, mu_init, scale_init];
    end

    % Parameter bounds
    % For parameters:
    %   0 < alpha <= 2
    %   -1 <= beta <= 1
    %   scale > 0
    bound.lo = [1e-3, -1, -Inf, 1e-3];
    bound.hi = [2, 1, Inf, Inf];
    bound.which = [1, 1, 0, 1]; % Indicate which parameters are constrained

    % Optimization options
    nobs = length(x);
    maxiter = length(initvec) * 100;
    tol = 1e-8;
    MaxFunEvals = length(initvec) * 400;
    opts = optimset('Display', 'none', 'MaxIter', maxiter, 'TolFun', tol, 'TolX', tol, 'MaxFunEvals', MaxFunEvals, 'LargeScale', 'Off');

    % Use fminunc for optimization
    [pout, fval, exitflag, output, grad, hess] = fminunc(@(param) StableNegLogLik(param, x, bound), ...
        einschrk(initvec, bound), opts);

    % Compute variance-covariance matrix
    Varcov = inv(hess) / nobs; % Inverse Hessian divided by number of observations
    [param, Varcov] = einschrk(pout, bound, Varcov); % Transform back parameters
    stderr = sqrt(diag(Varcov)); % Standard errors
    loglik = -fval * nobs; % Log-likelihood at optimum
    iters = output.iterations; % Number of iterations
end

function nll = StableNegLogLik(param, x, bound)
    % Apply constraints
    if isstruct(bound)
        paramvec = einschrk(real(param), bound, 999);
    else
        paramvec = param;
    end

    alpha = paramvec(1);
    beta = paramvec(2);
    mu = paramvec(3);
    scale = paramvec(4);

    % Constraints: 0 < alpha <= 2, -1 <= beta <= 1, scale > 0
    if alpha <= 0 || alpha > 2 || abs(beta) > 1 || scale <= 0
        nll = 1e5;
        return;
    end

    % Standardize data
    z = (x - mu) / scale;

    % Compute the log-likelihood
    pdf_values = asymstabpdf(z, alpha, beta); % Call to PDF function
    if any(pdf_values <= 0)
        nll = 1e5;
        return;
    end
    ll = log(pdf_values) - log(scale); % Include Jacobian adjustment
    nll = -mean(ll); % Negative log-likelihood
end

function [f] = asymstabpdf(xvec, alpha, beta)
    % asymstabpdf Computes the PDF of the asymmetric stable Paretian distribution.
    %
    % Inputs:
    %   xvec  - Data vector
    %   alpha - Stability parameter (0 < alpha <= 2)
    %   beta  - Skewness parameter (-1 <= beta <= 1)
    %
    % Output:
    %   f     - PDF values

    if alpha == 1
        error('PDF calculation for alpha = 1 is not supported.');
    end

    f = zeros(size(xvec));
    for i = 1:length(xvec)
        x = xvec(i);
        if x < 0
            f(i) = stab(-x, alpha, -beta);
        else
            f(i) = stab(x, alpha, beta);
        end
    end
end

function pdf = stab(x, alpha, beta)
    % stab Computes the PDF of the stable distribution for a single value.
    t0 = (1 / alpha) * atan(beta * tan(pi * alpha / 2));
    if x == 0
        % Special case for x = 0 (Borak et al., 2005)
        pdf = gamma(1 + (1 / alpha)) * cos(t0) / (pi * (1 + (beta * tan(pi * alpha / 2))^2)^(1 / (2 * alpha)));
    else
        tol = 1e-9;
        integ = quadv(@(t) integrand(t, abs(x), alpha, t0), -t0, pi / 2, tol);
        pdf = alpha * x^(1 / (alpha - 1)) / (pi * abs(alpha - 1)) * integ;
    end
end

function I = integrand(t, x, alpha, t0)
    % integrand Computes the integrand for the stable distribution PDF.
    ct = cos(t);
    s = t0 + t;
    v = (cos(alpha * t0))^(1 / (alpha - 1)) * (ct / sin(alpha * s))^(alpha / (alpha - 1)) * ...
        (cos(alpha * s - t) / ct);
    term = -x^(alpha / (alpha - 1));
    I = v .* exp(term * v);
end



