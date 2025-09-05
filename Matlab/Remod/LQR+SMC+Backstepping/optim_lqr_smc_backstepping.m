clc;
clear;
close all;

%% Define Plant Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % length of the pendulum [m]
r   = 0.0726;     % wheel radius [m]
g   = 9.81;       % gravitational acceleration [m/s^2]

% Derived parameters from the linearized model:
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;  % determinant

%% State-Space Model
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

%% LQR Controller (Fixed)
% These Q and R matrices are preset.
Q = diag([100, 1, 1000, 1]);  
R = 0.01; 
K = lqr(A, B, Q, R);

%% GA Optimization Setup for Combined Control Strategy
% Decision vector: p = [α, β, γ, λ, η, ε, k1, k2, k3, k4]
% Bounds:
lb = [0, 0, 0, 0.1, 0.1, 0.001, 0, 0, 0, 0];
ub = [1, 1, 1, 10, 20, 0.1, 20, 20, 20, 20];

options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',400);

% The cost function simulates the closed-loop response and returns an integrated error.
costFun = @(p) costFunctionCombined(p, A, B, K, 0.001, 5);

% Run GA optimization
[p_opt, fval] = ga(costFun, 10, [], [], [], [], lb, ub, [], options);

fprintf('\nOptimized Combined Control Parameters:\n');
fprintf('α = %.3f, β = %.3f, γ = %.3f\n', p_opt(1), p_opt(2), p_opt(3));
fprintf('λ = %.3f, η = %.3f, ε = %.4f\n', p_opt(4), p_opt(5), p_opt(6));
fprintf('k1 = %.3f, k2 = %.3f, k3 = %.3f, k4 = %.3f\n', p_opt(7), p_opt(8), p_opt(9), p_opt(10));

%% Simulate Combined Control with Optimized Parameters
[t_sim, x_sim, u_sim] = simulateCombinedControl(p_opt, A, B, K, 0.001, 5);

figure;
subplot(2,1,1);
plot(t_sim, x_sim(1,:), 'b', 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('Combined Control Response (Optimized)');
grid on;
subplot(2,1,2);
plot(t_sim, x_sim(3,:), 'r', 'LineWidth', 1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

figure;
plot(t_sim(1:end-1), u_sim, 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Control Input (N)');
title('Filtered Control Effort (Optimized)');
grid on;

%% --- Function Definitions ---

function cost = costFunctionCombined(p, A, B, K, ts, T_end)
    % p = [α, β, γ, λ, η, ε, k1, k2, k3, k4]
    alpha = p(1); beta = p(2); gamma = p(3);
    lambda = p(4); eta = p(5); epsilon = p(6);
    k1 = p(7); k2 = p(8); k3 = p(9); k4 = p(10);
    
    t = 0:ts:T_end;
    N = length(t);
    x = zeros(4, N);
    x(:,1) = [0; 0; 0.1; 0];  % initial conditions (small pendulum deviation)
    u = zeros(1, N-1);
    filtered_u = 0;
    alpha_filter = 0.05;  % low-pass filter coefficient
    
    % Simulation via Euler integration
    for i = 1:N-1
        % Current state
        x1 = x(1,i); x2 = x(2,i); x3 = x(3,i); x4 = x(4,i);
        % LQR control
        u_lqr = -K * x(:,i);
        % Sliding Mode Control (SMC)
        s = x3 + lambda*x4;
        u_smc = -eta * tanh(s / epsilon);
        % Backstepping Control
        v1 = -k1*x3;
        v2 = -k2*x4;
        v3 = -k3*(x1 + x2);
        v4 = -k4*(x3 + x4);
        u_bs = v1 + v2 + v3 + v4;
        % Combined control
        u_raw = alpha*u_lqr + beta*u_bs + gamma*u_smc;
        filtered_u = (1 - alpha_filter)*filtered_u + alpha_filter*u_raw;
        u(i) = filtered_u;
        
        % Update state (Euler integration)
        dx = A*x(:,i) + B*u(i);
        x(:,i+1) = x(:,i) + ts*dx;
    end
    
    % Performance metric: Integrated absolute error for cart position and pendulum angle.
    t_sim = t;  % time vector
    e_cart = abs(x(1,:) - 0);  % desired cart position is 0
    e_pend = abs(x(3,:) - 0);  % desired pendulum angle is 0
    cost = trapz(t_sim, e_cart) + trapz(t_sim, e_pend);
    
    % Heavy penalty if cost is unreasonable
    if isnan(cost) || isinf(cost)
        cost = 1e6;
    end
end

function [t, x, u] = simulateCombinedControl(p, A, B, K, ts, T_end)
    t = 0:ts:T_end;
    N = length(t);
    x = zeros(4, N);
    x(:,1) = [0; 0; 0.1; 0];
    u = zeros(1, N-1);
    filtered_u = 0;
    alpha_filter = 0.05;
    
    alpha = p(1); beta = p(2); gamma = p(3);
    lambda = p(4); eta = p(5); epsilon = p(6);
    k1 = p(7); k2 = p(8); k3 = p(9); k4 = p(10);
    
    for i = 1:N-1
        x1 = x(1,i); x2 = x(2,i); x3 = x(3,i); x4 = x(4,i);
        u_lqr = -K * x(:,i);
        s = x3 + lambda*x4;
        u_smc = -eta * tanh(s / epsilon);
        v1 = -k1*x3;
        v2 = -k2*x4;
        v3 = -k3*(x1 + x2);
        v4 = -k4*(x3 + x4);
        u_bs = v1 + v2 + v3 + v4;
        u_raw = alpha*u_lqr + beta*u_bs + gamma*u_smc;
        filtered_u = (1 - alpha_filter)*filtered_u + alpha_filter*u_raw;
        u(i) = filtered_u;
        dx = A*x(:,i) + B*u(i);
        x(:,i+1) = x(:,i) + ts*dx;
    end
end
