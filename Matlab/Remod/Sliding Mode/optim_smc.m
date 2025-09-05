clc;
clear all;
close all;

%% Plant Parameters
m_w = 0.432;      % Mass of each wheel [kg]
m_p = 5.0;        % Mass of the pendulum [kg]
I_w = 0.5 * m_w * (0.0726)^2;  % Moment of inertia of each wheel [kg*m^2]
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

%% SMC Design Parameters to be Optimized
% Decision vector: p = [lambda1, lambda2, lambda3, K, phi]
% Bounds:
%   lambda1: [1, 20]
%   lambda2: [1, 50]
%   lambda3: [0.1, 5]
%   K      : [1, 100]
%   phi    : [0.001, 0.2]
lb = [1, 1, 0.1, 1, 0.001];
ub = [20, 50, 5, 100, 0.2];

options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',100);

% Cost function handle: it simulates the SMC closed-loop response and computes cost.
costFun = @(p) costFunctionSMC(p, A, B, a, b, c, d, Delta);

[p_opt, fval] = ga(costFun, 5, [], [], [], [], lb, ub, [], options);

fprintf('\nOptimized SMC parameters:\n');
fprintf('lambda1 = %.3f, lambda2 = %.3f, lambda3 = %.3f, K = %.3f, phi = %.3f\n', ...
    p_opt(1), p_opt(2), p_opt(3), p_opt(4), p_opt(5));

%% Use the Optimized Parameters to Simulate the SMC
params.a = a;
params.b = b;
params.c = c;
params.d = d;
params.Delta = Delta;
params.lambda1 = p_opt(1);
params.lambda2 = p_opt(2);
params.lambda3 = p_opt(3);
params.K = p_opt(4);
params.phi = p_opt(5);

tspan = [0 5];            % Simulation time span [s]
x0    = [0; 0; 0.1; 0];    % Initial condition: small pendulum angle deviation

[t, x] = ode45(@(t, x) smc_composite_dynamics(t, x, A, B, params), tspan, x0);

figure;
subplot(2,1,1);
plot(t, x(:,1), 'b', 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('Nonlinear SMC Response with Optimized Parameters');
grid on;

subplot(2,1,2);
plot(t, x(:,3), 'r', 'LineWidth', 1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

%% --- Function Definitions ---

function cost = costFunctionSMC(p, A, B, a, b, c, d, Delta)
    % p = [lambda1, lambda2, lambda3, K, phi]
    lambda1 = p(1);
    lambda2 = p(2);
    lambda3 = p(3);
    K       = p(4);
    phi     = p(5);
    
    % Pack parameters for simulation
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
    
    % Simulation settings
    tspan = [0, 5];
    x0 = [0; 0; 0.1; 0];
    
    % Run simulation using ODE45
    [t, x] = ode45(@(t, x) smc_composite_dynamics(t, x, A, B, params), tspan, x0);
    
    % Define a tolerance for settling (for both cart position and pendulum angle)
    tol = 0.02;
    T_set_cart = settlingTime(t, x(:,1), tol);
    T_set_pend = settlingTime(t, x(:,3), tol);
    
    % Cost is defined as the sum of the settling times.
    cost = T_set_cart + T_set_pend;
    
    % Penalize if the simulation does not settle properly.
    if isnan(cost) || isinf(cost)
        cost = 1e6;
    end
end

function T_set = settlingTime(t, signal, tol)
    % Compute settling time as the last time the error (from the final value) exceeds tol.
    final_value = signal(end);
    err = abs(signal - final_value);
    idx = find(err > tol);
    if isempty(idx)
        T_set = t(1);
    else
        T_set = t(idx(end));
    end
    % If the settling time is near the simulation end, penalize.
    if T_set >= t(end)-0.1
        T_set = 1e3;
    end
end

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
    
    % Saturation function (with boundary layer to reduce chattering)
    sat_val = min(max(s/phi, -1), 1);
    
    % Compute the equivalent part of s_dot (without the control input)
    s_dot_equiv = lambda1*x(2) + lambda2*x(4) + (d/Delta)*(-b + lambda3*a)*x(3);
    
    % Effective gain factor: k_u = (c - lambda3*b)/Delta, where c is passed via params
    k_u = (params.c - lambda3*b)/Delta;
    
    % Control law: u = (1/k_u)*(- s_dot_equiv - K*sat(s/phi))
    u = (1/k_u)*(- s_dot_equiv - K*sat_val);
    
    % Closed-loop dynamics
    dx = A*x + B*u;
end
