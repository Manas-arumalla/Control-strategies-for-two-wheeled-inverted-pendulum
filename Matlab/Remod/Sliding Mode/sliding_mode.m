clc;
clear all;

%% System Parameters
m_w = 0.432;      % Mass of each wheel [kg]
m_p = 5.0;        % Mass of the pendulum [kg]
I_w = 1/2 * m_w * (0.0726)^2;  % Moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;             % Moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % Length of the pendulum [m]
r   = 0.0726;     % Wheel radius [m]
g   = 9.81;       % Gravitational acceleration [m/s^2]

a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;

A = [0, 1, 0, 0;
     0, 0, -b*d/Delta, 0;
     0, 0, 0, 1;
     0, 0, a*d/Delta, 0];
B = [0;
     c/Delta;
     0;
    -b/Delta];

%% SMC Design Parameters 
% Define the sliding surface as a combination of states:
% s = lambda1*x1 + x2 + lambda2*x3 + lambda3*x4
lambda1 = 5;    % Gain for cart position
lambda2 = 20;   % Gain for pendulum angle
lambda3 = 1.001;    % Gain for pendulum angular velocity
K       = 25;   % Sliding mode control gain (tune for performance/chattering)
phi     = 0.05; % Boundary layer thickness for saturation function

% For linearized analysis, we compute the effective state-feedback gain.
% The sliding surface: s = L*x, where L = [lambda1, 1, lambda2, lambda3]
L = [lambda1, 1, lambda2, lambda3];

% The equivalent part from s_dot (without the u term) is:
% s_dot_equiv = lambda1*x2 + lambda2*x4 + (d/Delta)*(-b+lambda3*a)*x3
L_eq = [0, lambda1, d/Delta*(-b + lambda3*a), lambda2];

% Effective gain factor: k_u = (c - lambda3*b)/Delta
k_u = (c - lambda3*b)/Delta;

% Thus the effective state-feedback gain from the SMC is:
K_eff = (1/k_u)*(L_eq + (K/phi)*L);

%% Compute Closed-Loop A Matrix for the Linearized Equivalent System
A_cl = A - B*K_eff;

% Define output matrix C and direct feedthrough D.
% Here we choose two outputs: cart position (x1) and pendulum angle (x3)
C = [1, 0, 0, 0;
     0, 0, 1, 0];
D = [0; 0];

%% Simulate the linear SMC Using ODE45
% Pack parameters into a structure for use in the dynamics function.
params.a = a;
params.b = b;
params.c = c;
params.d = d;
params.Delta = Delta;
params.lambda1 = lambda1;
params.lambda2 = lambda2;
params.lambda3 = lambda3;
params.K = K;
params.phi = phi;

tspan = [0 5];            % Simulation time span [s]
x0    = [0; 0; 0.1; 0];    % Initial condition: small pendulum angle deviation

[t, x] = ode45(@(t, x) smc_composite_dynamics(t, x, A, B, params), tspan, x0);

%% Plot Nonlinear Simulation Results
figure;
subplot(2,1,1);
plot(t, x(:,1), 'b', 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('SMC Response');
grid on;

subplot(2,1,2);
plot(t, x(:,3), 'r', 'LineWidth', 1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
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

% %% Linearized Closed-Loop Analysis for Frequency-Domain Plots
% sys_cl = ss(A_cl, B, C, D);
% 
% % For frequency-domain analysis, we consider the pendulum angle output (second output)
% sys_tf_pendulum = tf(sys_cl(2));
% 
% % Root Locus
% figure;
% rlocus(sys_tf_pendulum);
% title('Root Locus (Pendulum Angle)');
% grid on;
% 
% % Nyquist Plot
% figure;
% nyquist(sys_tf_pendulum);
% title('Nyquist Plot (Pendulum Angle)');
% grid on;
% 
% % Bode Plot
% figure;
% bode(sys_tf_pendulum);
% title('Bode Plot (Pendulum Angle)');
% grid on;

%% --- Function Definitions ---
function dx = smc_composite_dynamics(~, x, A, B, params)
    % Unpack parameters
    a = params.a;
    b = params.b;
    Delta = params.Delta;
    d = params.d;
    lambda1 = params.lambda1;
    lambda2 = params.lambda2;
    lambda3 = params.lambda3;
    K = params.K;
    phi = params.phi;
    
    % Composite sliding surface: s = lambda1*x1 + x2 + lambda2*x3 + lambda3*x4
    s = lambda1*x(1) + x(2) + lambda2*x(3) + lambda3*x(4);
    
    % Saturation function to reduce chattering (using a boundary layer)
    sat_val = min(max(s/phi, -1), 1);
    
    % Compute the equivalent part of s_dot (without the control term)
    % s_dot_equiv = lambda1*x2 + lambda2*x4 + (d/Delta)*(-b+lambda3*a)*x3
    s_dot_equiv = lambda1*x(2) + lambda2*x(4) + d/Delta*(-b + lambda3*a)*x(3);
    
    % Effective gain factor: k_u = (c - lambda3*b)/Delta, where c is passed via params
    k_u = (params.c - lambda3*b)/Delta;
    
    % Control law: u = (1/k_u)*(- s_dot_equiv - K*sat(s/phi))
    u = (1/k_u)*(- s_dot_equiv - K*sat_val);
    
    % Closed-loop dynamics
    dx = A*x + B*u;
end
