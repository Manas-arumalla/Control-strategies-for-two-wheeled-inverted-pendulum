%% EPSAC Implementation with GA-Based Parameter Optimization
% This script uses a genetic algorithm to optimize key EPSAC design
% parameters for an inverted pendulum on a cart.
%
% Decision vector:
%    p = [Np, Nc, lambda_control]
% where:
%   Np: Prediction Horizon (steps, integer in [5,50])
%   Nc: Control Horizon (steps, integer in [1,20], with Nc <= Np)
%   lambda_control: Weight on control effort (continuous in [0.001,1])
%
% The cost function simulates the closed-loop EPSAC controller over the
% simulation horizon and returns the sum of squared tracking errors.

clear; clc; close all;

%% Define Plant Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % length of the pendulum [m]
r   = 0.0726;     % wheel radius [m]
g   = 9.81;       % gravitational acceleration [m/s^2]

% Derived parameters from the linearized model:
%   a*x_ddot + b*theta_ddot = F
%   b*x_ddot + c*theta_ddot = d*theta
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;  % determinant

%% Construct the Linearized State–Space Model
% States: x = [cart_position; cart_velocity; pendulum_angle; pendulum_angular_velocity]
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

sys_ss = ss(A, B, C, D);

%% Discretize the System
Ts = 0.1;  % Sampling time (seconds)
sys_d = c2d(sys_ss, Ts);

%% EPSAC Fixed Settings (besides parameters to be optimized)
ref = [0; 0];  % Stabilization: reference is zero for both outputs.
u_min = -10;
u_max = 10;

T_sim = 10;                    % Simulation time (s)
N_sim = T_sim / Ts;            % Number of simulation steps

%% GA Optimization Setup
% Decision vector: p = [Np, Nc, lambda_control]
% Bounds:
%   Np: integer in [5, 50]
%   Nc: integer in [1, 20]
%   lambda_control: continuous in [0.001, 1]
lb = [5, 1, 0.001];
ub = [50, 20, 1];
IntCon = [1, 2];  % Np and Nc must be integers

options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
costFun = @(p) costFunctionEPSAC(p, sys_d, ref, Ts, N_sim, u_min, u_max, A, B, C);

% Nonlinear constraint to enforce Nc <= Np.
nonlcon = @(p) nonlconEPSAC(p);

[p_opt, fval] = ga(costFun, 3, [], [], [], [], lb, ub, nonlcon, IntCon, options);

fprintf('\nOptimized EPSAC Parameters:\n');
fprintf('Prediction Horizon (Np) = %d\n', round(p_opt(1)));
fprintf('Control Horizon (Nc)    = %d\n', round(p_opt(2)));
fprintf('Control Weight (lambda) = %.4f\n', p_opt(3));

%% Simulate EPSAC with Optimized Parameters
Np_opt = round(p_opt(1));
Nc_opt = round(p_opt(2));
lambda_opt = p_opt(3);

% Initialize the estimated model (adaptive update) as the nominal model.
A_est = sys_d.A;
B_est = sys_d.B;

x = zeros(4, N_sim+1);         % State trajectory
x(:,1) = [0; 0; 0.1; 0];      % Initial state with small deviation
u_store = zeros(1, N_sim);     % Control inputs
y_store = zeros(2, N_sim);     % Outputs
time = (0:N_sim-1)*Ts;         % Time vector

for k = 1:N_sim
    xk = x(:,k);
    % Construct the reference trajectory over the prediction horizon.
    R_traj = repmat(ref, 1, Np_opt);
    
    % Optimize future control moves (only for Nc_opt moves).
    U0 = zeros(Nc_opt,1);  % initial guess
    opt_options = optimoptions('fmincon','Display','off','Algorithm','sqp');
    [U_opt, ~] = fmincon(@(U) costFunction(U, xk, R_traj, A_est, B_est, Nc_opt, Np_opt, C, lambda_opt), ...
                          U0, [], [], [], [], ...
                          u_min*ones(Nc_opt,1), u_max*ones(Nc_opt,1), [], opt_options);
    
    % Apply the first control input.
    u_k = U_opt(1);
    u_store(k) = u_k;
    
    % Update the state using the discretized model.
    x(:,k+1) = sys_d.A * xk + sys_d.B * u_k;
    y_store(:,k) = C * xk;
    
    % Simple adaptation update for estimated model.
    adaptation_gain = 0.001;
    pred_x = A_est * xk + B_est * u_k;
    error_x = x(:,k+1) - pred_x;
    A_est = A_est + adaptation_gain * error_x * xk';
    B_est = B_est + adaptation_gain * error_x * u_k;
end

%% Plot the Results
figure;
subplot(2,1,1);
plot(time, x(1,1:end-1), 'b','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Cart Position');
title('Optimized EPSAC: Cart Position vs. Time');
grid on;

subplot(2,1,2);
plot(time, x(3,1:end-1), 'r','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
title('Optimized EPSAC: Pendulum Angle vs. Time');
grid on;

figure;
plot(time, u_store, 'k','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Control Input');
title('Optimized EPSAC: Control Input vs. Time');
grid on;

%% --- Function Definitions ---
function J = costFunctionEPSAC(p, sys_d, ref, Ts, N_sim, u_min, u_max, A, B, C)
    % p = [Np, Nc, lambda_control]
    Np = round(p(1));
    Nc = round(p(2));
    lambda = p(3);
    
    % Initialize the estimated model as the nominal discretized model.
    A_est = sys_d.A;
    B_est = sys_d.B;
    
    % Initialize state trajectory.
    x = zeros(4, N_sim+1);
    x(:,1) = [0.1; 0; 0.1; 0];
    J = 0;
    
    % Run EPSAC simulation over N_sim steps.
    for k = 1:N_sim
        xk = x(:,k);
        R_traj = repmat(ref, 1, Np);
        U0 = zeros(Nc,1);
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');
        % Use the inner cost function for the control sequence.
        [U_opt, ~] = fmincon(@(U) costFunction(U, xk, R_traj, A_est, B_est, Nc, Np, C, lambda), U0, [], [], [], [], ...
                              u_min*ones(Nc,1), u_max*ones(Nc,1), [], options);
        u_k = U_opt(1);
        x(:,k+1) = sys_d.A*xk + sys_d.B*u_k;
        
        % Accumulate cost as squared tracking error (for both outputs).
        yk = C*xk;
        e = ref - yk;
        J = J + sum(e.^2);
        
        % Update estimated model using a simple adaptation law.
        adapt_gain = 0.001;
        pred_x = A_est*xk + B_est*u_k;
        error_x = x(:,k+1) - pred_x;
        A_est = A_est + adapt_gain * error_x * xk';
        B_est = B_est + adapt_gain * error_x * u_k;
    end
end

function J = costFunction(U, x0, R, A_est, B_est, Nc, Np, C, lambda)
    % Extend the control sequence to cover the full prediction horizon.
    U_ext = [U; repmat(U(end), Np - Nc, 1)];
    x_pred = x0;
    J = 0;
    for i = 1:Np
        x_pred = A_est*x_pred + B_est*U_ext(i);
        y_pred = C*x_pred;
        e = R(:,i) - y_pred;
        J = J + (e' * e) + lambda*(U_ext(i)^2);
    end
end

function [c, ceq] = nonlconEPSAC(p)
    % Ensure that the control horizon Nc is not greater than the prediction horizon Np.
    Np = round(p(1));
    Nc = round(p(2));
    c = Nc - Np;  % Constraint: Nc - Np <= 0.
    ceq = [];
end
