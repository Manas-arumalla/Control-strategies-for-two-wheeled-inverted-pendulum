%% Koopman Eigenfunctions + LQR with GA-based Tuning for Two-Wheeled Inverted Pendulum
% This script demonstrates how to use a genetic algorithm to tune five LQR
% parameters (four weights for the physical states in Q and R) for minimizing
% the integrated absolute pendulum angle error.
%
% The procedure is as follows:
% 1. Data collection using EDMD to compute the Koopman operator.
% 2. GA optimization: for each candidate set of parameters, an LQR gain is computed 
%    (in the lifted space) and a closed-loop simulation is run. The cost is defined 
%    as the integrated absolute error of the pendulum angle.
% 3. The GA penalizes candidate parameters that yield a non-stabilizing LQR (by 
%    catching errors from dlqr).
%
% Written by [Your Name], [Date]

clear; clc; close all;

%% 1. Define System Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

% Derived coefficients (from the derivation)
a = 2*m_w + m_p + 2*I_w/(r^2);  % effective horizontal inertia
b = m_p * l;                    % coupling coefficient
c = m_p * l^2 + I_p;            % pendulum inertia

%% 2. Data Collection for EDMD (Koopman Operator Approximation)
Ntraj = 30;      % number of trajectories
Tsim  = 5;       % simulation time per trajectory (s)
dt    = 0.02;    % sampling time-step
time  = 0:dt:Tsim;
nSteps = length(time);

% Define lifted state dimension
nLift = 10;
Psi_data = [];   % each column is psi(x_k)
Psi_next = [];   % each column is psi(x_{k+1})
U_data   = [];   % corresponding control inputs

rng(0); % for reproducibility
for traj = 1:Ntraj
    % Generate a random initial condition (small deviations)
    x0 = [0.1*randn; 0.1*randn; 0.1*randn; 0.1*randn];
    
    % Generate a small random control signal for excitation
    U_traj = 0.5*randn(1, nSteps);
    
    % Simulate the nonlinear dynamics using Euler integration for data collection
    x_traj = zeros(4, nSteps);
    x_traj(:,1) = x0;
    for k = 1:nSteps-1
        x_dot = nonlinearDynamics(x_traj(:,k), U_traj(k), a, b, c, m_p, g, l);
        x_traj(:,k+1) = x_traj(:,k) + dt*x_dot;
    end
    
    % Collect lifted data
    for k = 1:nSteps-1
        psi_k  = liftState(x_traj(:,k));
        psi_kp = liftState(x_traj(:,k+1));
        Psi_data = [Psi_data, psi_k];
        Psi_next = [Psi_next, psi_kp];
        U_data   = [U_data, U_traj(k)]; %#ok<AGROW>
    end
end

%% 3. EDMD: Compute Koopman Matrices
% Solve for matrices A_koop and B_koop such that:
%    psi(x_{k+1}) ≈ A_koop * psi(x_k) + B_koop * u_k
DataMat = [Psi_data; U_data];
KoopmanAB = Psi_next * pinv(DataMat);
A_koop = KoopmanAB(:, 1:nLift);
B_koop = KoopmanAB(:, end);

%% 4. Genetic Algorithm for LQR Parameter Tuning
% We wish to optimize 5 parameters:
%    X = [q1, q2, q3, q4, R]
% where the Q matrix is defined as:
%    Q = diag([q1, q2, q3, q4, 50, 50, 10, 10, 10, 1])
% and R is a scalar.
%
% The objective is based on the integrated absolute error in pendulum angle.
objFun = @(X) closedLoopCost(X, A_koop, B_koop, a, b, c, m_p, g, l);

% Parameter bounds:
% For q1 and q3 (weights on x and theta): [1, 500]
% For q2 and q4 (velocities): [0.1, 50]
% For R: [0.1, 100]
lb = [1, 0.1, 1, 0.1, 0.1];
ub = [500, 50, 500, 50, 100];

% GA options
options = optimoptions('ga','Display','iter','PopulationSize',20);

nvars = 5;
[bestParams, bestCost] = ga(objFun, nvars, [], [], [], [], lb, ub, [], options);

fprintf('Optimized parameters:\n');
fprintf('q1 = %.2f, q2 = %.2f, q3 = %.2f, q4 = %.2f, R = %.2f\n', bestParams);
fprintf('Best cost (integrated |theta| error) = %.4f\n', bestCost);

%% 5. Closed-Loop Simulation with Optimized Parameters
% Build Q and R from the optimized parameters
Q_opt = diag([bestParams(1), bestParams(2), bestParams(3), bestParams(4), 50, 50, 10, 10, 10, 1]);
R_opt = bestParams(5);
[K_koop_opt,~,~] = dlqr(A_koop, B_koop, Q_opt, R_opt);

% Closed-loop simulation settings
x0 = [0; 0; 0.1; 0];  % initial state: cart at 0, pendulum angle 0.1 rad, zero velocities
Tsim_cl = 10;        % simulation time (s)
optionsODE = odeset('RelTol',1e-6, 'AbsTol',1e-8);

[t_cl, x_cl] = ode15s(@(t,x) closedLoopDynamics(t, x, a, b, c, m_p, g, l, K_koop_opt), [0 Tsim_cl], x0, optionsODE);

%% 6. Plot the Optimized Closed-Loop Results
figure;
subplot(2,1,1);
plot(t_cl, x_cl(:,1), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Cart Position x (m)');
title('Optimized: Cart Position under Koopman-based LQR');

subplot(2,1,2);
plot(t_cl, x_cl(:,3), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Pendulum Angle \theta (rad)');
title('Optimized: Pendulum Angle under Koopman-based LQR');

figure;
% Compute control input over time
u_opt = zeros(length(t_cl),1);
for i = 1:length(t_cl)
    u_val = -K_koop_opt * liftState(x_cl(i,:)');
    max_u = 100;
    u_opt(i) = max(min(u_val, max_u), -max_u);
end
plot(t_cl, u_opt, 'LineWidth',2);
xlabel('Time (s)'); ylabel('Control Input F (N)');
title('Optimized: Control Input');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% -------------------- Local Functions -------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cost = closedLoopCost(X, A_koop, B_koop, a, b, c, m_p, g, l)
    % closedLoopCost runs a closed-loop simulation using LQR gains computed from
    % candidate parameters X = [q1, q2, q3, q4, R] and returns a cost.
    %
    % The Q matrix is constructed as:
    %    Q = diag([q1, q2, q3, q4, 50, 50, 10, 10, 10, 1])
    % and R is the fifth element.
    %
    % The cost is defined as the integrated absolute error in pendulum angle.
    
    Q = diag([X(1), X(2), X(3), X(4), 50, 50, 10, 10, 10, 1]);
    R = X(5);
    
    % Compute LQR gain for the lifted system.
    try
        [K_koop,~,~] = dlqr(A_koop, B_koop, Q, R);
    catch
        cost = 1e6;  % Penalize infeasible parameters
        return;
    end
    
    % Set initial condition and simulation parameters.
    x0 = [0; 0; 0.1; 0];
    Tsim = 10;
    optionsODE = odeset('RelTol',1e-6, 'AbsTol',1e-8);
    
    % Run the closed-loop simulation.
    try
        [t, x] = ode15s(@(t,x) closedLoopDynamics(t, x, a, b, c, m_p, g, l, K_koop), [0 Tsim], x0, optionsODE);
    catch
        cost = 1e6;
        return;
    end
    
    % Define the cost as the integrated absolute error in pendulum angle.
    theta = x(:,3);
    cost = trapz(t, abs(theta));
    
    % Penalize if the cost is NaN or infinite.
    if isnan(cost) || isinf(cost)
        cost = 1e6;
    end
end

function xdot = nonlinearDynamics(x, F, a, b, c, m_p, g, l)
    % nonlinearDynamics computes the state derivative for the two-wheeled inverted pendulum.
    % State: x = [x; x_dot; theta; theta_dot]
    % F: control force.
    theta = x(3);
    theta_dot = x(4);
    Delta = a*c - b^2*cos(theta)^2; % theta-dependent denominator
    x_ddot = ( c*(F + b*sin(theta)*theta_dot^2) - b*cos(theta)*m_p*g*l*sin(theta) ) / Delta;
    theta_ddot = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*(F + b*sin(theta)*theta_dot^2) ) / Delta;
    xdot = [ x(2); x_ddot; theta_dot; theta_ddot ];
end

function psi = liftState(x)
    % liftState returns the dictionary of observables for state x.
    % Here, psi(x) = [ x; x_dot; theta; theta_dot; sin(theta); cos(theta);
    %                   x^2; theta^2; x*theta; 1 ]
    psi = [ x(1);
            x(2);
            x(3);
            x(4);
            sin(x(3));
            cos(x(3));
            x(1)^2;
            x(3)^2;
            x(1)*x(3);
            1 ];
end

function dxdt = closedLoopDynamics(~, x, a, b, c, m_p, g, l, K_koop)
    % closedLoopDynamics computes the closed-loop state derivative.
    % The control is computed as u = -K_koop*psi(x) with saturation.
    z = liftState(x);
    u = -K_koop * z;
    max_u = 100;
    u = max(min(u, max_u), -max_u);
    dxdt = nonlinearDynamics(x, u, a, b, c, m_p, g, l);
end
