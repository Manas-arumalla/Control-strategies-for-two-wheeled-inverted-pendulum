%% 
% Hybrid Control Architecture Overview
% 
% The controller is designed as a three–layer cascade combining nonlinear compensation, adaptive learning, and predictive optimization to robustly stabilize a two–wheeled inverted pendulum system.
% 
% Outer–Loop NMPC (Nonlinear Model Predictive Control)
% 
% Purpose:
% The NMPC layer optimizes the control effort over a finite future horizon while handling constraints on the actuator.
% Function:
% It uses a linearized state–space model of the system to predict future states. At every simulation step, it minimizes a quadratic cost function (penalizing state deviations and control energy) using the interior–point algorithm. Only the first control input from the optimized sequence is applied.
% Role in the Architecture:
% NMPC provides a baseline control signal that ensures overall system performance and constraint adherence.
% Inner–Loop DSC (Dynamic Surface Control)
% 
% Purpose:
% DSC is used to cancel the dominant nonlinearities of the system without the complexity of higher–order derivative computations.
% Function:
% It defines an error based on the pendulum angle (deviation from the upright position) and filters the error derivative using a first–order filter. A corrective control law is then applied (scaled by a gain) to rapidly reduce the error.
% Role in the Architecture:
% This layer acts as a fast–acting compensator to stabilize the inner–loop dynamics, ensuring that the system responds quickly to perturbations.
% Adaptive NN (Neural Network) Augmentation
% 
% Purpose:
% The adaptive NN layer learns and compensates for residual modeling uncertainties and unmodeled dynamics that remain after DSC.
% Function:
% A simple neural network, represented here by a scalar weight, estimates an extra control term based on the error signal. The weight is updated online using a gradient descent rule, and anti–windup measures (clamping) ensure stability.
% Role in the Architecture:
% By providing adaptive compensation, this layer enhances robustness and adjusts the control signal in real time to account for discrepancies between the model and actual system behavior.
% Combined Control Input and Saturation
% 
% The total control input is obtained by summing the NMPC, DSC, and NN components.
% A saturation limit is applied to the final control input to prevent excessively large commands that might destabilize the system.
% Summary
% 
% NMPC optimizes control over a horizon, ensuring constraint compliance and overall performance.
% DSC rapidly compensates for the nonlinear dynamics of the pendulum, stabilizing the inner loop.
% Adaptive NN learns residual errors and refines the control input, improving robustness.
% Control Saturation ensures that the final command remains within practical limits.
% This multi–layered, hybrid approach leverages the strengths of predictive optimization, fast nonlinear compensation, and adaptive learning to achieve robust stabilization of the inverted pendulum system.
%%
clc; clear; close all;

%% System Parameters
m_w = 0.432;           % Mass of each wheel [kg]
m_p = 5.0;             % Mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % Inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;       % Inertia of the pendulum [kg*m^2]
l   = 0.4;             % Distance from axle to pendulum COM [m]
r   = 0.0726;          % Wheel radius [m]
g   = 9.81;            % Gravity [m/s^2]

% Coefficients from the dynamics derivation:
a = 2*m_w + m_p + 2*I_w/(r^2);    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

%% Linearized Model (for NMPC design)
Delta = a*c - b^2;
A_lin = [ 0,     1,                      0,              0;
          0,     0,               -b*d/Delta,              0;
          0,     0,                      0,              1;
          0,     0,                a*d/Delta,              0];
      
B_lin = [ 0;
          c/Delta;
          0;
         -b/Delta];
     
C_lin = [1, 0, 0, 0; 
         0, 0, 1, 0];
D_lin = [0; 0];

%% Simulation Parameters
ts = 0.001;          % Time step [s]
T_final = 5;         % Final simulation time [s]
t = 0:ts:T_final;
N = length(t);
x = zeros(4, N);
% Initial state: [cart position; cart velocity; pendulum angle; angular velocity]
x(:,1) = [0; 0; 0.1; 0];  

% Preallocate control input storage
u_total = zeros(1, N);

%% Controller Tuning Parameters

% ----- Inner-Loop DSC (Dynamic Surface Control) -----
lambda_dsc = 3.0;      % DSC filter coefficient
alpha_dsc  = 20;       % DSC gain
z = 0;                 % DSC filter state initialization

% ----- Adaptive NN Augmentation -----
nn_w = 0;              % Initial NN weight (scalar example)
gamma_nn = 0.05;       % Reduced learning rate for NN weight update
nn_w_max = 5;          % NN weight clamping limits
nn_w_min = -5;

% ----- Outer-Loop NMPC -----
N_horizon = 10;        % Prediction horizon (number of steps)
Q_nmpc = diag([10, 1, 100, 1]); % State weighting matrix
R_nmpc = 0.01;         % Input weighting (scalar)
% Use interior-point algorithm for fmincon
options = optimoptions('fmincon','Display','none','Algorithm','interior-point');

% ----- Control Saturation -----
u_sat = 10;  % Maximum absolute control input [N]

%% Main Simulation Loop
for k = 1:N-1
    % Current measured state
    x_meas = x(:,k);
    
    % ----- Outer-Loop NMPC -----
    u0 = zeros(N_horizon,1); % Initial guess for NMPC control sequence
    cost_fun = @(u_seq) nmpc_cost(u_seq, x_meas, A_lin, B_lin, Q_nmpc, R_nmpc, N_horizon);
    u_min = -u_sat*ones(N_horizon,1);
    u_max =  u_sat*ones(N_horizon,1);
    
    % Solve NMPC using fmincon
    u_seq_opt = fmincon(cost_fun, u0, [], [], [], [], u_min, u_max, [], options);
    u_nmpc = u_seq_opt(1);  % Use only the first control input

    % ----- Inner-Loop DSC -----
    e = x_meas(3);      % Pendulum angle error (desired = 0)
    e_dot = x_meas(4);  
    % Update DSC filter state (Euler integration)
    z = (1 - ts*lambda_dsc)*z + ts*e_dot;
    u_dsc = -alpha_dsc * (e + z);
    
    % ----- Adaptive NN Augmentation -----
    u_nn = nn_w * e;
    % Update NN weight (gradient descent on e^2)
    nn_w = nn_w + gamma_nn * e^2;
    % Clamp the NN weight to prevent windup
    nn_w = max(min(nn_w, nn_w_max), nn_w_min);
    
    % ----- Combine Control Inputs and Saturate -----
    u_combined = u_nmpc + u_dsc + u_nn;
    % Apply saturation to the total control input
    u_total(k) = max(min(u_combined, u_sat), -u_sat);
    
    % ----- Full Nonlinear Dynamics Simulation -----
    theta = x_meas(3);
    theta_dot = x_meas(4);
    denom = a*c - b^2*cos(theta)^2;
    ddx = ( c*( u_total(k) + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*(m_p*g*l*sin(theta)) ) / denom;
    ddtheta = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*( u_total(k) + b*sin(theta)*theta_dot^2 ) ) / denom;
    
    x_dot = [ x_meas(2);
              ddx;
              x_meas(4);
              ddtheta ];
    % Euler integration for state update
    x(:,k+1) = x_meas + ts * x_dot;
end

%% Plot Results
figure;
subplot(2,1,1);
plot(t, x(1,:), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('Hybrid Control Response: DSC + Adaptive NN + NMPC');
grid on;

subplot(2,1,2);
plot(t, x(3,:), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

figure;
plot(t(1:end-1), u_total(1:end-1), 'k', 'LineWidth', 1.5);
ylabel('Control Input (N)');
xlabel('Time (s)');
title('Combined Control Effort');
grid on;

%% NMPC Cost Function (Subfunction)
function J = nmpc_cost(u_seq, x0, A, B, Q, R, N_horizon)
    % Ensure u_seq is a column vector and pad to N_horizon if needed
    u_seq = u_seq(:);
    if length(u_seq) < N_horizon
        u_seq = [u_seq; zeros(N_horizon - length(u_seq),1)];
    end
    x_pred = x0;
    J = 0;
    for k = 1:N_horizon
        u = u_seq(k);
        x_pred = A*x_pred + B*u;
        J = J + x_pred'*Q*x_pred + R*(u^2);
    end
    if ~isfinite(J)
        J = 1e6;
    end
end
