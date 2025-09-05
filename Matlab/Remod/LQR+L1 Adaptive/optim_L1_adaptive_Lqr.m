clc; clear; close all;

%% System Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * m_w * (0.0726)^2;  % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;             % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % Length of the pendulum [m]
r   = 0.0726;     % wheel radius [m]
g   = 9.81;       % acceleration due to gravity [m/s^2]

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

% Disturbance input matrix (applied to x4 dynamics; set to zero here)
E = [0; 0; 0; 0];

%% Nominal LQR Controller Design (for reference)
% (Not used directly in GA, but shows typical values.)
Q_nom = diag([1000, 1, 100, 1]);
R_nom = 0.01;
K_nom = lqr(A, B, Q_nom, R_nom);
A_cl_nom = A - B*K_nom;
P_nom = lyap(A_cl_nom', eye(4));

%% GA Optimization Setup for Combined L1 Adaptive and LQR Tuning
% Decision vector: p = [Gamma, tau, sigma_max, q1, q2, q3, q4, r_val]
% Bounds:
%   Gamma: [1, 100]
%   tau: [0.01, 0.2]
%   sigma_max: [1, 50]
%   q1: [1, 5000]
%   q2: [1, 100]
%   q3: [1, 1000]
%   q4: [1, 100]
%   r_val: [0.001, 1]
lb = [1, 0.01, 1, 1, 1, 1, 1, 0.001];
ub = [100, 0.2, 50, 5000, 100, 1000, 100, 1];

% External disturbance (e.g., a sinusoid) applied to the plant (if desired)
d_ext = @(t) 0.5*sin(0.5*t);

% Total simulation time for evaluation
Tfinal = 8;

options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
costFun = @(p) costFunctionL1(p, A, B, E, d_ext, Tfinal);

[p_opt, fval] = ga(costFun, 8, [], [], [], [], lb, ub, [], options);

fprintf('\nOptimized Parameters:\n');
fprintf('Gamma = %.3f, tau = %.4f, sigma_max = %.3f\n', p_opt(1), p_opt(2), p_opt(3));
fprintf('q1 = %.3f, q2 = %.3f, q3 = %.3f, q4 = %.3f, R = %.3f\n', p_opt(4), p_opt(5), p_opt(6), p_opt(7), p_opt(8));

%% Simulate with Optimized Parameters
Gamma = p_opt(1);
tau = p_opt(2);
sigma_max = p_opt(3);
q1 = p_opt(4); q2 = p_opt(5); q3 = p_opt(6); q4 = p_opt(7); r_val = p_opt(8);

% Compute LQR gain with optimized Q and R:
Q_opt = diag([q1, q2, q3, q4]);
R_opt = r_val;
K_opt = lqr(A, B, Q_opt, R_opt);
A_cl_opt = A - B*K_opt;
P_opt = lyap(A_cl_opt', eye(4));

dt = 0.001;
tspan = 0:dt:Tfinal;

% Initial conditions for augmented state: [x; x_hat; sigma_hat; z_f]
x0 = [0; 0; 0.1; 0];
x_hat0 = zeros(4,1);
sigma_hat0 = 0;
z_f0 = 0;
z0 = [x0; x_hat0; sigma_hat0; z_f0];

options_ode = odeset('RelTol',1e-6, 'AbsTol',1e-9);
[t, Z] = ode45(@(t,z) augmentedDynamics(t, z, A, B, E, K_opt, Gamma, tau, P_opt, d_ext, sigma_max), tspan, z0, options_ode);

x_sim = Z(:, 1:4);
sigma_hat_sim = Z(:, 9);
z_f_sim = Z(:, 10);
u_sim = - (x_sim * K_opt') - z_f_sim;

figure;
subplot(2,1,1);
plot(t, x_sim(:,1), 'b', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Cart Position');
title('Optimized Combined Control: Cart Position');
grid on;
subplot(2,1,2);
plot(t, x_sim(:,3), 'r', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Pendulum Angle');
title('Optimized Combined Control: Pendulum Angle');
grid on;

figure;
plot(t, u_sim, 'k', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Control Input');
title('Optimized Combined Control: Control Effort');
grid on;

figure;
plot(t, sigma_hat_sim, 'm', 'LineWidth',1.5); hold on;
plot(t, z_f_sim, 'c--', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Adaptive Terms');
title('Optimized Combined Control: Adaptive Estimate and Filtered Term');
legend('\sigma_{hat}','z_f');
grid on;

%% --- Function Definitions ---

function cost = costFunctionL1(p, A, B, E, d_ext, Tfinal)
    % p = [Gamma, tau, sigma_max, q1, q2, q3, q4, r_val]
    Gamma = p(1);
    tau = p(2);
    sigma_max = p(3);
    q1 = p(4); q2 = p(5); q3 = p(6); q4 = p(7);
    r_val = p(8);
    
    Q = diag([q1, q2, q3, q4]);
    R = r_val;
    
    % Compute LQR gain with candidate Q and R.
    try
        K = lqr(A, B, Q, R);
    catch
        cost = 1e6;
        return;
    end
    A_cl = A - B*K;
    
    % Compute Lyapunov matrix for adaptation law.
    try
        P = lyap(A_cl', eye(4));
    catch
        cost = 1e6;
        return;
    end
    
    % Simulation settings
    dt = 0.001;
    tspan = 0:dt:Tfinal;
    % Initial augmented state: [x; x_hat; sigma_hat; z_f]
    x0 = [0; 0; 0.1; 0];
    x_hat0 = zeros(4,1);
    sigma_hat0 = 0;
    z_f0 = 0;
    z0 = [x0; x_hat0; sigma_hat0; z_f0];
    
    try
        options_ode = odeset('RelTol',1e-6, 'AbsTol',1e-9);
        [t, Z] = ode45(@(t,z) augmentedDynamics(t, z, A, B, E, K, Gamma, tau, P, d_ext, sigma_max), tspan, z0, options_ode);
        x = Z(:, 1:4);
        % Use the pendulum angle (x3) for settling time measurement.
        signal = x(:,3);
        tol = 0.02;
        T_set = settlingTime(t, signal, tol);
        cost = T_set;
        if T_set >= t(end)-0.1 || isnan(T_set)
            cost = 1e6;
        end
    catch
        cost = 1e6;
    end
end

function T_set = settlingTime(t, signal, tol)
    final_val = signal(end);
    err = abs(signal - final_val);
    idx = find(err > tol);
    if isempty(idx)
        T_set = t(1);
    else
        T_set = t(idx(end));
    end
    if T_set >= t(end)-0.1
        T_set = 1e3;
    end
end

function dz = augmentedDynamics(t, z, A, B, E, K, Gamma, tau, P, d_ext, sigma_max)
    % Unpack augmented state: z = [x; x_hat; sigma_hat; z_f]
    x = z(1:4);
    x_hat = z(5:8);
    sigma_hat = z(9);
    z_f = z(10);
    
    % Compute control input: u = -K*x - z_f
    u = -K * x - z_f;
    
    % Plant dynamics with external disturbance applied via E.
    x_dot = A*x + B*u + E*d_ext(t);
    % Predictor dynamics.
    x_hat_dot = A*x_hat + B*(u + sigma_hat);
    % Adaptation law with projection:
    sigma_dot_temp = Gamma * (B' * P * (x - x_hat));
    if sigma_hat >= sigma_max && sigma_dot_temp > 0
        sigma_hat_dot = 0;
    elseif sigma_hat <= -sigma_max && sigma_dot_temp < 0
        sigma_hat_dot = 0;
    else
        sigma_hat_dot = sigma_dot_temp;
    end
    % Low-pass filter dynamics.
    z_f_dot = (-1/tau)*z_f + (1/tau)*sigma_hat;
    
    dz = [x_dot; x_hat_dot; sigma_hat_dot; z_f_dot];
end
