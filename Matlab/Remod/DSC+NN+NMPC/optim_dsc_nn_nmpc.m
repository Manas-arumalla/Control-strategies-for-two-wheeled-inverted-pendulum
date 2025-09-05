%% Optimize Hybrid Controller Parameters via GA
% This script optimizes the tuning parameters for a hybrid controller that
% combines NMPC, DSC, and Adaptive NN augmentation for an inverted pendulum on a cart.
%
% Decision vector:
%    p = [N_horizon, Q1, Q2, Q3, Q4, R_nmpc, lambda_dsc, alpha_dsc, gamma_nn, nn_w_max, nn_w_min]
%
% where:
%   - N_horizon: NMPC prediction horizon (steps, integer)
%   - Q1–Q4: Diagonal entries for the NMPC state weighting matrix
%   - R_nmpc: NMPC input weighting (scalar)
%   - lambda_dsc: DSC filter coefficient
%   - alpha_dsc: DSC gain
%   - gamma_nn: Learning rate for adaptive NN weight update
%   - nn_w_max, nn_w_min: Clamping limits for the NN weight
%
% The cost function simulates the closed-loop system (using Euler integration)
% over T_final seconds and returns the integrated squared tracking error (of cart position and pendulum angle).
% If any error occurs in the inner optimization, a high cost is returned.

clear; clc; close all;

%% 1. Model Parameters
m_w = 0.432;           % mass of each wheel [kg]
m_p = 5.0;             % mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % inertia of the pendulum [kg*m^2]
l   = 0.4;             % distance from axle to pendulum COM [m]
r   = 0.0726;          % wheel radius [m]
g   = 9.81;            % gravitational acceleration [m/s^2]

% Derived parameters
a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

%% 2. Linearized Model for NMPC
Delta = a*c - b^2;
A_lin = [ 0,     1,                      0,              0;
          0,     0,               -b*d/Delta,              0;
          0,     0,                      0,              1;
          0,     0,                a*d/Delta,              0];
B_lin = [0; c/Delta; 0; -b/Delta];
C_lin = [1, 0, 0, 0; 0, 0, 1, 0];
D_lin = [0; 0];

%% 3. Simulation Settings
ts = 0.001;          % time step [s]
T_final = 10;        % simulation time [s]
N_sim = T_final/ts;  
x0 = [0; 0; 0.1; 0];  % initial state [cart pos; cart vel; pendulum angle; angular velocity]

%% 4. GA Decision Vector and Bounds
% p = [N_horizon, Q1, Q2, Q3, Q4, R_nmpc, lambda_dsc, alpha_dsc, gamma_nn, nn_w_max, nn_w_min]
% Suggested bounds:
%   N_horizon: [5, 20] (integer)
%   Q1: [1, 100]       (weight on x)
%   Q2: [0.1, 50]      (weight on x_dot)
%   Q3: [1, 500]       (weight on theta)
%   Q4: [0.1, 50]      (weight on theta_dot)
%   R_nmpc: [0.001, 1]
%   lambda_dsc: [1, 5]
%   alpha_dsc: [5, 50]
%   gamma_nn: [0.001, 0.1]
%   nn_w_max: [1, 10]
%   nn_w_min: [-10, -1]
lb = [5,   1,    0.1,   1,   0.1, 0.001,  1,   5,   0.001,  1, -10];
ub = [20, 100,   50,  500,   50,    1,     5,  50,   0.1,   10, -1];
IntCon = 1;  % Only N_horizon is an integer.

%% 5. GA Optimization Setup
options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
costFun = @(p) hybridCostFunction(p, A_lin, B_lin, C_lin, T_final, ts, x0);
[p_opt, fval] = ga(costFun, 11, [], [], [], [], lb, ub, [], IntCon, options);

fprintf('\nOptimized Hybrid Controller Parameters:\n');
fprintf('N_horizon = %d\n', round(p_opt(1)));
fprintf('Q = diag([%.2f, %.2f, %.2f, %.2f])\n', p_opt(2), p_opt(3), p_opt(4), p_opt(5));
fprintf('R_nmpc = %.4f\n', p_opt(6));
fprintf('lambda_dsc = %.2f, alpha_dsc = %.2f\n', p_opt(7), p_opt(8));
fprintf('gamma_nn = %.4f\n', p_opt(9));
fprintf('nn_w_max = %.2f, nn_w_min = %.2f\n', p_opt(10), p_opt(11));

%% 6. Final Simulation with Optimized Parameters
N_horizon = round(p_opt(1));
Q_nmpc = diag(p_opt(2:5));
R_nmpc = p_opt(6);
lambda_dsc = p_opt(7);
alpha_dsc = p_opt(8);
gamma_nn = p_opt(9);
nn_w_max = p_opt(10);
nn_w_min = p_opt(11);

% Initialize estimated model (for NMPC adaptation) as the nominal linear model.
A_est = A_lin;
B_est = B_lin;

x = zeros(4, N_sim+1);
x(:,1) = x0;
u_total = zeros(1, N_sim);
% Initialize DSC filter state and NN weight.
z_filter = 0;
nn_w_current = 0;

for k = 1:N_sim
    xk = x(:,k);
    
    % ----- Outer-Loop NMPC -----
    % Construct reference trajectory over N_horizon (stabilization: zero reference)
    ref = zeros(2, N_horizon);
    U0 = zeros(N_horizon,1);
    nmpc_options = optimoptions('fmincon','Display','off','Algorithm','sqp');
    try
        cost_nmpc = @(U) innerCostFunction(U, xk, ref, A_est, B_est, N_horizon, C_lin, lambda_dsc);
        U_opt = fmincon(cost_nmpc, U0, [], [], [], [], -10*ones(N_horizon,1), 10*ones(N_horizon,1), [], nmpc_options);
        u_nmpc = U_opt(1);
    catch
        % If fmincon fails, apply a default control (or high cost)
        u_nmpc = 0;
    end
    
    % ----- Inner-Loop DSC -----
    e = xk(3);       % pendulum angle error (desired = 0)
    e_dot = xk(4);
    z_filter = (1 - ts*lambda_dsc)*z_filter + ts*e_dot;
    u_dsc = -alpha_dsc*(e + z_filter);
    
    % ----- Adaptive NN Augmentation -----
    u_nn = nn_w_current * e;
    nn_w_current = nn_w_current + gamma_nn * e^2;
    nn_w_current = max(min(nn_w_current, nn_w_max), nn_w_min);
    
    % ----- Combine Control Inputs and Saturate -----
    u_combined = u_nmpc + u_dsc + u_nn;
    u_total(k) = max(min(u_combined, 10), -10);
    
    % ----- Update Plant State (simulate nonlinear dynamics) -----
    theta = xk(3);
    theta_dot = xk(4);
    denom = a*c - b^2*cos(theta)^2;
    ddx = ( c*( u_total(k) + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*( m_p*g*l*sin(theta) ) ) / denom;
    ddtheta = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*( u_total(k) + b*sin(theta)*theta_dot^2 ) ) / denom;
    x_dot = [ xk(2);
              ddx;
              theta_dot;
              ddtheta ];
    x(:,k+1) = xk + ts*x_dot;
    
    % ----- Update Estimated Model (simple adaptation) -----
    adapt_gain = 0.001;
    pred_x = A_est*xk + B_est*u_total(k);
    error_x = x(:,k+1) - pred_x;
    A_est = A_est + adapt_gain * error_x * xk';
    B_est = B_est + adapt_gain * error_x * u_total(k);
end

%% Plot Final Results
time = 0:ts:T_final;
figure;
subplot(3,1,1);
plot(time, x(1,1:end-1),'b','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Cart Position (m)');
title('Optimized Hybrid Controller: Cart Position');
grid on;

subplot(3,1,2);
plot(time, x(3,1:end-1),'r','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
title('Optimized Hybrid Controller: Pendulum Angle');
grid on;

subplot(3,1,3);
plot(time, u_total,'k','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Control Input (N)');
title('Optimized Hybrid Controller: Control Input');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Subfunction: Inner Cost Function for NMPC ---
function J = innerCostFunction(U, x0, R_traj, A_est, B_est, Np, C, lambda)
    U_ext = U;
    if length(U) < Np
        U_ext = [U; repmat(U(end), Np - length(U), 1)];
    end
    x_pred = x0;
    J = 0;
    for i = 1:Np
        x_pred = A_est*x_pred + B_est*U_ext(i);
        y_pred = C*x_pred;
        e = R_traj(:,i) - y_pred;
        J = J + (e' * e) + lambda*(U_ext(i)^2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Subfunction: Hybrid Controller Cost Function ---
function J = hybridCostFunction(p, A_lin, B_lin, C_lin, T_final, ts, x0)
    % p = [N_horizon, Q1, Q2, Q3, Q4, R_nmpc, lambda_dsc, alpha_dsc, gamma_nn, nn_w_max, nn_w_min]
    Np = round(p(1));
    Q_vals = p(2:5);
    R_nmpc = p(6);
    lambda_dsc = p(7);
    alpha_dsc = p(8);
    gamma_nn = p(9);
    nn_w_max = p(10);
    nn_w_min = p(11);
    
    Q = diag(Q_vals);
    R = R_nmpc;
    
    % Initialize estimated model for NMPC.
    A_est = A_lin;
    B_est = B_lin;
    
    N_sim = T_final/ts;
    x = zeros(4, N_sim+1);
    x(:,1) = x0;
    J = 0;
    
    u_min = -10;
    u_max = 10;
    
    for k = 1:N_sim
        xk = x(:,k);
        ref = zeros(2, Np);  % zero reference
        U0 = zeros(Np,1);
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');
        % Wrap fmincon call in try-catch to avoid undefined objective errors.
        try
            [U_opt, ~] = fmincon(@(U) innerCostFunction(U, xk, ref, A_est, B_est, Np, C_lin, lambda_dsc), ...
                                  U0, [], [], [], [], u_min*ones(Np,1), u_max*ones(Np,1), [], options);
            u_k = U_opt(1);
        catch
            u_k = 0;
        end
        x(:,k+1) = A_lin*xk + B_lin*u_k;
        yk = C_lin*xk;
        e = -yk;  % reference is zero
        J = J + sum(e.^2);
        
        % Update estimated model
        adapt_gain = 0.001;
        pred_x = A_est*xk + B_est*u_k;
        error_x = x(:,k+1) - pred_x;
        A_est = A_est + adapt_gain * error_x * xk';
        B_est = B_est + adapt_gain * error_x * u_k;
    end
    if ~isfinite(J)
        J = 1e6;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Dummy Nonlinear Constraint ---
function [c, ceq] = nonlconEPSAC(p)
    c = -1; ceq = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
