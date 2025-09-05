%% GA Optimization for ADRC Adaptive Controller Gains
clc;
clear;
close all;

%% System Parameters
m_w = 0.432;           % Mass of each wheel [kg]
m_p = 5.0;             % Mass of the pendulum [kg]
I_w = 0.5 * m_w * (0.0726)^2; % Inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;       % Inertia of the pendulum [kg*m^2]
l = 0.4;               % Distance from axle to pendulum COM [m]
r = 0.0726;            % Wheel radius [m]
g = 9.81;              % Gravity [m/s^2]

% Derived coefficients
a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;

%% Simulation Parameters
dt = 0.001;
T_final = 10;
t_sim = 0:dt:T_final;

%% GA Optimization Setup
% Decision vector: p = [Kp_x, Kd_x, Kp_theta, Kd_theta]
% Set lower and upper bounds for the gains
lb = [0.1, 0.1, 0.1, 0.1];
ub = [100, 50, 200, 50];

% GA options (adjust PopulationSize and MaxGenerations as needed)
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50, 'MaxGenerations', 200);

% Cost function handle: the cost is defined as the sum of settling times for cart position and pendulum angle.
costFun = @(p) adrcCostFunction(p, m_w, m_p, I_w, I_p, l, r, g, a, b, c, d, Delta, dt, T_final);

% Run GA optimization.
[p_opt, fval] = ga(costFun, 4, [], [], [], [], lb, ub, [], options);

fprintf('\nOptimized Gains:\n');
fprintf('Kp_x     = %.3f\n', p_opt(1));
fprintf('Kd_x     = %.3f\n', p_opt(2));
fprintf('Kp_theta = %.3f\n', p_opt(3));
fprintf('Kd_theta = %.3f\n', p_opt(4));
fprintf('Cost (Tsett) = %.3f\n', fval);

%% Simulate Closed-Loop Response with Optimized Gains
% Observer gains (fixed)
w0 = 30;
L1 = 3*w0;
L2 = 3*w0^2;
L3 = w0^3;

% Use the optimal gains (do not apply adaptation here)
Kp_x = p_opt(1);
Kd_x = p_opt(2);
Kp_theta = p_opt(3);
Kd_theta = p_opt(4);

N = length(t_sim);
x = zeros(4, N);
x(:,1) = [0; 0; 0.1; 0]; % initial state: [cart position; cart velocity; pendulum angle; angular velocity]

% Initialize ESO estimates
z_x = x(1,1);
z_xdot = x(2,1);
z_theta = x(3,1);
z_thetadot = x(4,1);
z_dist_x = 0;
z_dist_theta = 0;

u_total = zeros(1, N);

for k = 1:N-1
    % Observation errors
    e_x = x(1,k) - z_x;
    e_theta = x(3,k) - z_theta;
    
    % ESO Updates
    z_x = z_x + dt*(z_xdot + L1*e_x);
    z_xdot = z_xdot + dt*(z_dist_x + L2*e_x);
    z_dist_x = z_dist_x + dt*(L3*e_x);
    
    z_theta = z_theta + dt*(z_thetadot + L1*e_theta);
    z_thetadot = z_thetadot + dt*(z_dist_theta + L2*e_theta);
    z_dist_theta = z_dist_theta + dt*(L3*e_theta);
    
    % Control errors (tracking zero)
    ex = -z_x;
    exdot = -z_xdot;
    etheta = -z_theta;
    ethetadot = -z_thetadot;
    
    % PD Control Laws
    u_x = Kp_x * ex + Kd_x * exdot;
    u_theta = Kp_theta * etheta + Kd_theta * ethetadot;
    
    % Combined Control Input (compensate for estimated disturbance)
    u = u_x + u_theta - (z_dist_x + z_dist_theta);
    
    % Control Saturation
    u = max(min(u, 10), -10);
    u_total(k) = u;
    
    % Dynamics Update
    theta = x(3,k);
    theta_dot = x(4,k);
    denom = a*c - b^2 * cos(theta)^2;
    ddx = ( c*( u + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*( m_p*g*l*sin(theta) ) ) / denom;
    ddtheta = ( a*( m_p*g*l*sin(theta) ) - b*cos(theta)*( u + b*sin(theta)*theta_dot^2 ) ) / denom;
    
    dx = [ x(2,k);
           ddx;
           x(4,k);
           ddtheta ];
    x(:,k+1) = x(:,k) + dt*dx;
end

%% Plot the Results
figure;
subplot(3,1,1);
plot(t_sim, x(1,:), 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('Cart Position with Optimized Gains');
grid on;

subplot(3,1,2);
plot(t_sim, x(3,:), 'LineWidth', 1.5);
ylabel('Pendulum Angle (rad)');
title('Pendulum Angle with Optimized Gains');
grid on;

subplot(3,1,3);
plot(t_sim, u_total, 'LineWidth', 1.5);
ylabel('Control Input (N)');
xlabel('Time (s)');
title('Control Input');
grid on;

%% --- Function Definitions ---
function cost = adrcCostFunction(p, m_w, m_p, I_w, I_p, l, r, g, a, b, c, d, Delta, dt, T_final)
    % p = [Kp_x, Kd_x, Kp_theta, Kd_theta]
    t = 0:dt:T_final;
    N = length(t);
    x = zeros(4, N);
    % Initial state: small initial pendulum deflection
    x(:,1) = [0; 0; 0.1; 0];
    
    % Observer Initialization (ESO)
    w0 = 30;
    L1 = 3*w0;
    L2 = 3*w0^2;
    L3 = w0^3;
    
    z_x = x(1,1);
    z_xdot = x(2,1);
    z_theta = x(3,1);
    z_thetadot = x(4,1);
    z_dist_x = 0;
    z_dist_theta = 0;
    
    % Use the controller gains from decision vector (no adaptation)
    Kp_x = p(1);
    Kd_x = p(2);
    Kp_theta = p(3);
    Kd_theta = p(4);
    
    for k = 1:N-1
        % ESO Updates
        e_x = x(1,k) - z_x;
        e_theta = x(3,k) - z_theta;
        
        z_x = z_x + dt*(z_xdot + L1*e_x);
        z_xdot = z_xdot + dt*(z_dist_x + L2*e_x);
        z_dist_x = z_dist_x + dt*(L3*e_x);
        
        z_theta = z_theta + dt*(z_thetadot + L1*e_theta);
        z_thetadot = z_thetadot + dt*(z_dist_theta + L2*e_theta);
        z_dist_theta = z_dist_theta + dt*(L3*e_theta);
        
        % Control computation (tracking zero)
        ex = -z_x;
        exdot = -z_xdot;
        etheta = -z_theta;
        ethetadot = -z_thetadot;
        
        u_x = Kp_x * ex + Kd_x * exdot;
        u_theta = Kp_theta * etheta + Kd_theta * ethetadot;
        u = u_x + u_theta - (z_dist_x + z_dist_theta);
        
        u = max(min(u, 10), -10);
        
        % Dynamics Update
        theta = x(3,k);
        theta_dot = x(4,k);
        denom = a*c - b^2 * cos(theta)^2;
        if abs(denom) < 1e-4
            cost = 1e6;
            return;
        end
        ddx = ( c*( u + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*( m_p*g*l*sin(theta) ) ) / denom;
        ddtheta = ( a*( m_p*g*l*sin(theta) ) - b*cos(theta)*( u + b*sin(theta)*theta_dot^2 ) ) / denom;
        
        dx = [ x(2,k); ddx; x(4,k); ddtheta ];
        x(:,k+1) = x(:,k) + dt*dx;
        
        % If the simulation becomes unstable, penalize the cost.
        if any(isnan(x(:,k+1))) || any(abs(x(:,k+1)) > 1e3)
            cost = 1e6;
            return;
        end
    end
    
    % Compute Settling Times for Cart Position (state 1) and Pendulum Angle (state 3)
    tol = 0.02;
    T_set_cart = computeSettlingTime(t, x(1,:), tol);
    T_set_theta = computeSettlingTime(t, x(3,:), tol);
    
    % Cost is the sum of the settling times (lower is better)
    cost = T_set_cart + T_set_theta;
    
    % Impose a heavy penalty if the settling times are too long.
    if T_set_cart >= T_final-0.1 || T_set_theta >= T_final-0.1
        cost = 1e6;
    end
end

function T_set = computeSettlingTime(t, signal, tol)
    % Compute settling time as the last time when the absolute error from the final value exceeds tol.
    final_value = signal(end);
    err = abs(signal - final_value);
    idx = find(err > tol);
    if isempty(idx)
        T_set = 0;
    else
        T_set = t(idx(end));
    end
end
