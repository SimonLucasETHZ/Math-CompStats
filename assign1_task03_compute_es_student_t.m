% MATLAB Script: plot_VaR_ES_t_distribution_corrected.m

% Clear workspace and close figures
clear; close all; clc;

% Parameters input
nu = input('Enter the degrees of freedom (nu): ');
while nu <= 0
    nu = input('Degrees of freedom must be positive. Please enter again: ');
end

xi = input('Enter the confidence level (xi, e.g., 0.95): ');
while xi <= 0 || xi >= 1
    xi = input('Confidence level must be between 0 and 1. Please enter again: ');
end

% Compute VaR
VaR = tinv(1 - xi, nu);

% Compute PDF at VaR
pdf_VaR = tpdf(VaR, nu);

% Corrected ES calculation using (1 - xi) in the denominator
ES = - ( ( nu + VaR^2 ) / ( ( nu - 1 ) * (1 - xi) ) ) * pdf_VaR;

% Display results
fprintf('\nValue at Risk (VaR) at confidence level %.2f: %.4f\n', xi, VaR);
fprintf('Expected Shortfall (ES) at confidence level %.2f: %.4f\n', xi, ES);

% Define the range for plotting
x_min = min(-5, ES - 1);
x_max = max(5, -VaR + 1);
x = linspace(x_min, x_max, 1000);

% Compute the PDF of the t-distribution
pdf_values = tpdf(x, nu);

% Create the plot
figure('Color', 'w', 'Position', [100, 100, 800, 600]);
plot(x, pdf_values, 'b', 'LineWidth', 2);
hold on;

% Shade the area corresponding to VaR and beyond
x_fill = linspace(x_min, VaR, 100);
y_fill = tpdf(x_fill, nu);
fill([x_fill, VaR], [y_fill, 0], 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

% Plot VaR line
plot([VaR, VaR], [0, tpdf(VaR, nu)], 'k--', 'LineWidth', 2);

% Plot ES line
plot([ES, ES], [0, tpdf(ES, nu)], 'g--', 'LineWidth', 2);

% Annotate VaR
text(VaR, tpdf(VaR, nu)*1.05, sprintf(' VaR = %.2f', VaR), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

% Annotate ES
text(ES, tpdf(ES, nu)*1.05, sprintf(' ES = %.2f', ES), 'FontSize', 12, 'Color', 'g', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

% Add labels and title
xlabel('Loss', 'FontSize', 14);
ylabel('Probability Density', 'FontSize', 14);

% Add legend
legend('t-Distribution PDF', 'VaR Region', 'VaR Threshold', 'ES Threshold', 'Location', 'Best');

% Grid and box
grid on;
box on;

% Adjust axes limits
xlim([x_min, x_max]);
ylim([0, max(pdf_values)*1.1]);

% Save the figure (optional)
% saveas(gcf, 'VaR_ES_t_distribution_plot.png');
