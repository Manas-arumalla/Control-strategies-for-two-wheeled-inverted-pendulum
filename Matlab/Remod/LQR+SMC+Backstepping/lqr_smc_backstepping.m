clc; clear; close all;

% System Parameters
m_w = 0.432;      % Mass of each wheel [kg]
m_p = 5.0;        % Mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;  % Moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % Moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % Wheel radius [m]
g   = 9.81;     % Gravity [m/s^2]

% State-Space Representation
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

C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

% LQR Controller
Q = diag([100, 1, 1000, 1]);  
R = 0.01; 
K = lqr(A, B, Q, R);

Ac = A - B*K;
sys_cl = ss(Ac, B, C, D);

% Sliding Mode Controller (SMC) Parameters
lambda = 3.5;   % Reduced sliding surface slope
eta = 5;        % Lower reaching gain
epsilon = 0.01; % Smoothness factor

% Backstepping Controller Gains (Tuned)
k1 = 8; k2 = 12; k3 = 6; k4 = 10;

% Control Weights
alpha = 0.7;  % Increased LQR influence
beta = 0.2;   % Backstepping weight
gamma = 0.1;  % Reduced SMC effect

% Simulation Parameters
ts = 0.001;
t = 0:ts:5;    
x = zeros(4, length(t));
x(:,1) = [0; 0; 0.1; 0];  % Initial conditions
u = zeros(1, length(t)); % Control input

% Filtering Variables
filtered_u = 0;
alpha_filter = 0.05;  % Low-pass filter coefficient

for i = 1:length(t)-1
    % Current state
    x1 = x(1,i); x2 = x(2,i); x3 = x(3,i); x4 = x(4,i);
    
    % Sliding Mode Control (SMC) with tanh smoothing
    s = x3 + lambda*x4; 
    u_smc = -eta * tanh(s / epsilon);
    
    % Backstepping Control
    v1 = -k1*x3;
    v2 = -k2*x4;
    v3 = -k3*(x1 + x2);
    v4 = -k4*(x3 + x4);
    u_bs = v1 + v2 + v3 + v4;
    
    % LQR Control
    u_lqr = -K * x(:,i);
    
    % Combined Control Strategy
    u_raw = alpha * u_lqr + beta * u_bs + gamma * u_smc;
    
    % Low-pass filter for smooth control effort
    filtered_u = (1 - alpha_filter) * filtered_u + alpha_filter * u_raw;
    u(i) = filtered_u;

    % Apply system dynamics (Euler integration)
    dx = A*x(:,i) + B*u(i);
    x(:,i+1) = x(:,i) + dx * ts;
    
    % Store control input
    nu(i) = u(i);
end

% Plot results
figure;
subplot(2,1,1);
plot(t, x(1,:), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('Combined Control Response');
grid on;

subplot(2,1,2);
plot(t, x(3,:), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

figure;
plot(t(1:end-1), nu, 'k', 'LineWidth', 1.5);
ylabel('Control Input (N)');
xlabel('Time (s)');
title('Filtered Control Effort');
grid on;

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = x(1,:);  % Cart/Wheel Position
angle_response = x(3,:);     % Pendulum Angle

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
