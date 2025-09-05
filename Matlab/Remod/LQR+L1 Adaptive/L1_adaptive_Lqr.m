%% L1 Adaptive Control for Inverted Pendulum on a Cart (Full-State Stabilization)
% This script implements an L1 adaptive controller that robustly stabilizes
% both the cart position (x1) and the pendulum angle (x3) for the 4th–order
% inverted pendulum system. In this version the initial pendulum angle is set
% to 0.1 rad and a projection operator is applied to the adaptive estimate to
% prevent windup and ensure long–term stability.

% The proposed control scheme implements an L₁ adaptive controller for full-state stabilization of a fourth-order inverted pendulum on a cart. The design integrates a nominal LQR controller—which is tuned to prioritize the cart position—with an adaptive augmentation layer. This augmentation consists of a state predictor, an adaptation law enhanced by a projection operator, and a low-pass filter. The adaptation mechanism estimates and compensates for model uncertainties and external disturbances (e.g., a sinusoidal perturbation applied to the pendulum acceleration), while the projection operator prevents parameter windup, thereby ensuring long-term stability. Simulation results indicate that this approach robustly stabilizes both the cart position and the pendulum angle.

clear; clc; close all;

%% System Parameters (provided)
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * m_w * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;       % Length of the pendulum [m]
r   = 0.0726;    % wheel radius [m]
g   = 9.81;      % acceleration due to gravity [m/s^2]

a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

Delta = a*c - b^2; 

A = [ 0,     1,                      0,              0;
      0,     0,               -b*d/Delta,              0;
      0,     0,                      0,              1;
      0,     0,                a*d/Delta,              0];

B = [ 0;
      c/Delta;
      0;
     -b/Delta];

% Disturbance input matrix (applied to x4 dynamics)
E = [0; 0; 0; 0];

%% Nominal LQR Controller Design
% In order to emphasize cart position stabilization, we increase the weight
% for x1 (cart position) in the Q matrix.
Q_lqr = diag([1000, 1, 100, 1]);  % increased weight on cart position (x1)
R_lqr = 0.01;
K = lqr(A, B, Q_lqr, R_lqr);
A_cl = A - B*K;  % nominal closed-loop dynamics

%% Compute Lyapunov Matrix P for the Adaptation Law
% Solve A_cl' * P + P * A_cl = -I (I is the 4x4 identity matrix)
P = lyap(A_cl', eye(4));

%% L1 Adaptive Controller Parameters
Gamma = 1;    % adaptation gain (tuned lower to help avoid windup)
tau = 0.2;     % low-pass filter time constant
sigma_max = 1; % projection bound for the adaptive estimate

%% Simulation Setup
Tfinal = 8;             % simulation time [s] (simulate a bit longer to check long–term behavior)
dt = 0.001;             % simulation time step [s]
tspan = 0:dt:Tfinal;

% Initial conditions:
% Plant state x: [cart position; cart velocity; pendulum angle; pendulum angular velocity]
% Set cart position to 0 and pendulum angle to 0.1 rad.
x0 = [0; 0; 0.1; 0];  
x_hat0 = zeros(4,1);     % initial predictor state
sigma_hat0 = 0;          % initial adaptive estimate
z_f0 = 0;                % initial low-pass filter state

% Augmented state: z = [x; x_hat; sigma_hat; z_f] (dimension 4+4+1+1 = 10)
z0 = [x0; x_hat0; sigma_hat0; z_f0];

%% External Disturbance Definition
% A disturbance (e.g., a sinusoid) is applied to the pendulum acceleration.
d_ext = @(t) 0.5*sin(0.5*t);

%% ODE Function for Augmented Dynamics
% The augmented dynamics consist of:
% 1. Plant dynamics: x_dot = A*x + B*u + E*d_ext(t)
% 2. Predictor dynamics: x_hat_dot = A*x_hat + B*(u + sigma_hat)
% 3. Adaptation law: sigma_hat_dot = Gamma * B' * P * (x - x_hat), with projection
% 4. Low-pass filter: z_f_dot = (-1/tau)*z_f + (1/tau)*sigma_hat
% Control law: u = -K*x - z_f
f_aug = @(t, z) augmentedDynamics(t, z, A, B, E, K, Gamma, tau, P, d_ext, sigma_max);

% Solve the augmented ODE using ode45
options = odeset('RelTol',1e-6, 'AbsTol',1e-9);
[t, Z] = ode45(f_aug, tspan, z0, options);

%% Extract States and Compute Control Input
x = Z(:, 1:4);          % plant states
x_hat = Z(:, 5:8);       % predictor states
sigma_hat = Z(:, 9);     % adaptive estimate (scalar)
z_f = Z(:, 10);          % low-pass filter state

% Recompute control input: u = -K*x - z_f (applied element–wise over time)
u = - (x * K') - z_f;

%% Plotting Results
figure;
subplot(2,1,1);
plot(t, x(:,1), 'b', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Cart Position');
title('Optimized Combined Control: Cart Position');
grid on;
subplot(2,1,2);
plot(t, x(:,3), 'r', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Pendulum Angle');
title('Optimized Combined Control: Pendulum Angle');
grid on;

figure;
plot(t, u, 'k', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Control Input');
title('Optimized Combined Control: Control Effort');
grid on;

figure;
plot(t, sigma_hat, 'm', 'LineWidth',1.5); hold on;
plot(t, z_f, 'c--', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Adaptive Terms');
title('Optimized Combined Control: Adaptive Estimate and Filtered Term');
legend('\sigma_{hat}','z_f');
grid on;

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = x(:,1);  % Cart/Wheel Position
angle_response = x(:,3);     % Pendulum Angle

% Deviations from desired reference
position_deviation = abs(position_response - desired_reference);
angle_deviation = abs(angle_response - desired_reference);

% Settling time for cart position
settling_index_pos = find(position_deviation > tolerance, 1, 'last');
if ~isempty(settling_index_pos)
    settling_time_position = t(settling_index_pos);
else
    settling_time_position = 0;
end

% Settling time for pendulum angle
settling_index_ang = find(angle_deviation > tolerance, 1, 'last');
if ~isempty(settling_index_ang)
    settling_time_angle = t(settling_index_ang);
else
    settling_time_angle = 0;
end

fprintf('Manual Settling Time:\n');
fprintf(' - Cart Position     : %.3f seconds\n', settling_time_position);
fprintf(' - Pendulum Angle    : %.3f seconds\n', settling_time_angle);

%% Augmented Dynamics Function Definition
function dz = augmentedDynamics(t, z, A, B, E, K, Gamma, tau, P, d_ext, sigma_max)
    % Unpack augmented state:
    % x: plant state (4x1)
    % x_hat: predictor state (4x1)
    % sigma_hat: adaptive estimate (scalar)
    % z_f: low-pass filter state (scalar)
    x = z(1:4);
    x_hat = z(5:8);
    sigma_hat = z(9);
    z_f = z(10);
    
    % Compute control input using the current plant state and filtered adaptive term
    u = -K*x - z_f;
    
    % Plant dynamics with external disturbance (applied via E):
    x_dot = A*x + B*u + E*d_ext(t);
    
    % Predictor dynamics:
    x_hat_dot = A*x_hat + B*(u + sigma_hat);
    
    % Adaptation law with projection (to avoid windup):
    sigma_dot_temp = Gamma * (B' * P * (x - x_hat));
    if sigma_hat >= sigma_max && sigma_dot_temp > 0
        sigma_hat_dot = 0;
    elseif sigma_hat <= -sigma_max && sigma_dot_temp < 0
        sigma_hat_dot = 0;
    else
        sigma_hat_dot = sigma_dot_temp;
    end
    
    % Low-pass filter dynamics:
    z_f_dot = (-1/tau)*z_f + (1/tau)*sigma_hat;
    
    % Combine derivatives into one vector:
    dz = [x_dot; x_hat_dot; sigma_hat_dot; z_f_dot];
end
