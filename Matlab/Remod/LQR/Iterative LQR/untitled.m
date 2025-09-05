clear; clc; close all;

A = [0 1 0 0;
     0 -0.0883167 0.629317 0;
     0 0 0 1;
     0 -0.235655 27.8285 0];

B = [0;
     0.883167;
     0;
     2.3565];

C = [0 0 1 0];

D = 0;


t = 0:0.01:50;
x0 = [0; 0; 1; 0];
desired_reference = 0;

k1 = 39.8;
k2 = 7.4;
k3 = 0;
k4 = -5;

u = @(x) -k1 * (x(3) - desired_reference) - k2 * x(4) - k3 * x(1) - k4 * x(2);

x = x0;
angle_response = zeros(length(t), 1);

for i = 1:length(t)
    u_val = u(x);
    x = x + (A * x + B * u_val) * 0.01;
    angle_response(i) = C * x;
end

figure;
plot(t, angle_response, 'LineWidth', 2);
title('Angle Response with Backstepping Controller', 'FontSize', 12);
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('Angle (radians)', 'FontSize', 12);
hold on;
yline(desired_reference, 'r--', 'Desired Reference', 'LineWidth', 1.5);
grid on;
legend('System Response', 'Desired Reference', 'Location', 'best');
xlim([0, 20]); 
ylim([-1, 1.5]); 

tolerance = 0.01;
settling_time_index = find(abs(angle_response - desired_reference) > tolerance, 1, 'last');
if ~isempty(settling_time_index)
    settling_time = t(settling_time_index);
else
    settling_time = 0;
end

fprintf('Settling time : %.2f seconds\n', settling_time);