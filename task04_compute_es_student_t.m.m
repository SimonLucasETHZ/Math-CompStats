% Task4_01.m

% Main code
nu = 5;  % Degrees of freedom
% Compute ES with default location, scale, and ESlevel (0.05)
ES = compute_ES_student_t(nu);

% Display the result
fprintf('Expected Shortfall (ES) with nu = %d: %.4f\n', nu, ES);

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
