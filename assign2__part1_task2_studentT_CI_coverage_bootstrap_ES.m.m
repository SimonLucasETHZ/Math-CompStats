

    % Example parameters
    df = 4;        % Degrees of freedom for Student's t-distribution
    n = 10000;     % Original data size
    B = 100;       % Number of bootstrap samples
    
    % Call the main function
    plot_ES_VaR_bootstrap(df, n, B);


function plot_ES_VaR_bootstrap(df, n, B)

        sim = 100;
        gamma = 0.01;    % Confidence level for VaR and ES (1%)

        if sim <= 0 || floor(sim) ~= sim
            error('Number of simulations sim must be a positive integer.');
        end

        coverage_studentt = zeros(sim, 1);
        coverage_nonparametric = zeros(sim, 1);
        length_studentt = zeros(sim, 1);
        length_nonparametric = zeros(sim, 1);

        ES_true = 0 + 1 * compute_ES_student_t(df, 0, 1, 0.01);
        VaR_true = 0 + 1 * tinv(0.01, 4);

        for s = 1:sim

        data = trnd(4, n, 1);  % Generate Student's t data (location=0, scale=1)
        fprintf('Starting %d simulation...\n', s);

           % -------------------------------
        CI_parametric = compute_parametric_bootstrap_single(data, B, n, gamma);
        % Record if true ES is within parametric CI
        coverage_studentt(s) = (CI_parametric(1) < ES_true) && (CI_parametric(2) > ES_true);
        % Record length of parametric CI
        length_studentt(s) = CI_parametric(2) - CI_parametric(1);
        
        % -------------------------------
        % Compute Non-Parametric Bootstrap CI
        % -------------------------------
        CI_nonparametric = compute_nonparametric_bootstrap_single(data, B, n, gamma);
        % Record if true ES is within non-parametric CI
        coverage_nonparametric(s) = (CI_nonparametric(1) < ES_true) && (CI_nonparametric(2) > ES_true);
        % Record length of non-parametric CI
        length_nonparametric(s) = CI_nonparametric(2) - CI_nonparametric(1);


        CI_mixture_of_normals = mixture_of_normals(data, B, n, gamma);
        coverage_mixture_of_normals(s) = (CI_mixture_of_normals(1) < ES_true) && (CI_mixture_of_normals(2) > ES_true);
        length_mixture_of_normals(s) = CI_mixture_of_normals(2) - CI_mixture_of_normals(1);

       
        end

         % -------------------------------
    % Compute Empirical Coverage and Average Lengths
    % -------------------------------
    empirical_coverage_parametric = mean(coverage_studentt) * 100;  % Convert to percentage
    empirical_coverage_nonparametric = mean(coverage_nonparametric) * 100;
    empirical_mixture_of_normals = mean(coverage_mixture_of_normals) * 100;

    avg_length_parametric = mean(length_studentt);
    avg_length_nonparametric = mean(length_nonparametric);
    avg_length_mixture_of_normals = mean(length_mixture_of_normals);
    
    % -------------------------------
    % Report Results
    % -------------------------------
    fprintf('Empirical Coverage (Parametric): %.2f%%\n', empirical_coverage_parametric);
    fprintf('Empirical Coverage (Non-Parametric): %.2f%%\n', empirical_coverage_nonparametric);
    fprintf('Empirical Coverage (Mixture of Normals): %.2f%%\n', empirical_mixture_of_normals);
    fprintf('Average CI Length (Parametric): %.4f\n', avg_length_parametric);
    fprintf('Average CI Length (Parametric): %.4f\n', avg_length_nonparametric);
    fprintf('Average CI Length (Mixture of Normals): %.4f\n', avg_length_mixture_of_normals);

    end
    
    %% -------------------------------
    % Bootstrap Functions
    % -------------------------------
    
    function CI_parametric = compute_parametric_bootstrap_single(data, B, n, gamma)
        ESlevel = gamma;                % ES confidence level
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
            VaR_values = location_mle + scale_mle * tinv(gamma, nu_mle);
            %ES_values(b) = location_mle - scale_mle * ((nu_mle + (tinv(gamma, nu_mle))^2) / (nu_mle - 1)) * tpdf(tinv(gamma, nu_mle), nu_mle) / gamma;    
            ES_values(b) =  mean(bootsamp(bootsamp <= VaR_values));  
        end

        lower_p = prctile(ES_values, 1);
        upper_p = prctile(ES_values, 99);
        CI_parametric = [lower_p, upper_p];
    end
    
    function CI_nonparametric = compute_nonparametric_bootstrap_single(data, B, n, gamma)
    
        VaRlevel = gamma;               % VaR confidence level
        ESlevel = gamma;                % ES confidence level
        ES_values = zeros(B, 1);        % Preallocate ES estimates
    
        for b = 1:B
            ind = unidrnd(n, [n, 1]);  % Bootstrap sample indices
            bootsamp = data(ind);        % Bootstrap sample
    
            VaR_bootstrap_np = quantile(bootsamp, VaRlevel);
            VaR_values = VaR_bootstrap_np;
    
            ES_values(b) = mean(bootsamp(bootsamp <= VaR_bootstrap_np));
        end
    % Compute 1th and 99th percentiles for 90% CI
    lower_np = prctile(ES_values, 1);
    upper_np = prctile(ES_values, 99);
    CI_nonparametric = [lower_np, upper_np];
    end
    
    function CI_mixture_of_normals = mixture_of_normals(data, B, n, gamma)
        ESlevel = gamma;                % ES confidence level
        ES_values = zeros(B, 1);        % Preallocate ES estimates
    
        for b = 1:B
            % Generate bootstrap sample indices
            ind = unidrnd(n, [n, 1]);  % Bootstrap sample indices
            bootsamp = data(ind);        % Bootstrap sample
            bootsamp = bootsamp(:);      % Ensure column vector
          %  try
                % Fit Gaussian Mixture Model with 2 components
                gm = fitgmdist(bootsamp, 2, 'Replicates', 5, 'Options', statset('MaxIter', 1000));
    
                % Generate samples from the fitted mixture model
                sample_mixnorm = random(gm, n);
    
                % Compute VaR for bootstrap sample
                VaR_bootstrap_p = quantile(sample_mixnorm, ESlevel);
                VaR_values = VaR_bootstrap_p;
    
                % Compute ES as mean of losses beyond VaR
                ES_values(b) = mean(sample_mixnorm(sample_mixnorm <= VaR_bootstrap_p));
            %catch ME
                % If fitting fails, assign NaN and optionally display a warning
             %   ES_values(b) = NaN;
              %  warning('GMM fitting failed on bootstrap sample %d: %s', b, ME.message);
           % end
        %end
        
    
        % Remove any NaN values resulting from failed fits
        valid_indices = ~isnan(VaR_values) & ~isnan(ES_values);
        ES_values = ES_values(valid_indices);

        lower_mix = prctile(ES_values, 1);
        upper_mix = prctile(ES_values, 99);
        CI_mixture_of_normals = [lower_mix, upper_mix];
        end
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
        % GAtsim Simulates data from the GAt distribution
        %
        % Inputs:
        %   sim   - Number of simulations
        %   d     - Parameter d
        %   v     - Parameter v
        %   theta - Parameter theta
        %
        % Output:
        %   x - Simulated data (sim x 1 vector)
    
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
    % NCT_quantile_estimation Estimates NCT parameters using quantile matching
    %
    % Inputs:
    %   data             - Data vector
    %   nct_lookup_table - Precomputed NCT quantiles table (optional)
    %   nu_grid          - Grid of nu values used in the table (optional)
    %   delta_grid       - Grid of delta values used in the table (optional)
    %
    % Outputs:
    %   nu_hat    - Estimated degrees of freedom
    %   delta_hat - Estimated noncentrality parameter
    %   mu_hat    - Estimated location parameter
    %   sigma_hat - Estimated scale parameter

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

