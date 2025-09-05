%% Koopman Eigenfunctions + LQR for Two-Wheeled Inverted Pendulum (Full Code)
% This script demonstrates a full workflow for implementing a Koopman-based
% LQR controller for a two-wheeled inverted pendulum system.
%
% The code includes:
% 1. System definition and parameter specification.
% 2. Data collection via Euler integration for EDMD (Extended Dynamic Mode Decomposition)
%    to approximate the Koopman operator.
% 3. EDMD to obtain a linear model in the lifted (Koopman) space.
% 4. LQR controller design in the lifted space.
% 5. Closed-loop simulation using a stiff solver (ode15s) with control saturation.
%
% Written by [Your Name], [Date]

clear; clc; close all;

%% 1. Define System Parameters
% Physical parameters for the wheels and the pendulum
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

% Derived coefficients (from your corrected derivation)
% a: effective horizontal inertia
a = 2*m_w + m_p + 2*I_w/(r^2);
% b: coupling coefficient (pendulum coupling)
b = m_p * l;
% c: pendulum inertia (includes the pendulum's moment of inertia)
c = m_p * l^2 + I_p;

%% 2. Data Collection for EDMD (Koopman Operator Approximation)
% Here we collect simulation data with small random control inputs so that we can
% build a linear model in the lifted (Koopman) space.
Ntraj = 30;      % number of trajectories
Tsim  = 5;       % simulation time for each trajectory (s)
dt    = 0.02;    % time-step for sampling
time  = 0:dt:Tsim;
nSteps = length(time);

% Define the dictionary dimension (lifted state dimension)
nLift = 10;
Psi_data = [];   % columns: psi(x_k)
Psi_next = [];   % columns: psi(x_{k+1})
U_data   = [];   % corresponding control inputs

rng(0); % for reproducibility
for traj = 1:Ntraj
    % Generate a random initial condition (small deviations)
    x0 = [0.1*randn; 0.1*randn; 0.1*randn; 0.1*randn];
    
    % Generate a small random control signal for the trajectory
    U_traj = 0.5*randn(1, nSteps);
    
    % Simulate the nonlinear dynamics using Euler integration for data collection
    x_traj = zeros(4, nSteps);
    x_traj(:,1) = x0;
    for k = 1:nSteps-1
        x_dot = nonlinearDynamics(x_traj(:,k), U_traj(k), a, b, c, m_p, g, l);
        x_traj(:,k+1) = x_traj(:,k) + dt*x_dot;
    end
    
    % Collect lifted data (dictionary observables) for each step
    for k = 1:nSteps-1
        psi_k = liftState(x_traj(:,k));
        psi_kp = liftState(x_traj(:,k+1));
        Psi_data = [Psi_data, psi_k];
        Psi_next = [Psi_next, psi_kp];
        U_data = [U_data, U_traj(k)]; %#ok<AGROW>
    end
end

%% 3. EDMD: Compute Koopman Matrices
% Solve for matrices A_koop and B_koop such that:
%      psi(x_{k+1}) ≈ A_koop * psi(x_k) + B_koop * u_k
DataMat = [Psi_data; U_data];
KoopmanAB = Psi_next * pinv(DataMat);
A_koop = KoopmanAB(:, 1:nLift);
B_koop = KoopmanAB(:, end);

%% 4. Design LQR Controller in the Lifted Space
% We design a discrete-time LQR controller for the lifted linear model:
%      z_{k+1} = A_koop*z + B_koop*u,  where z = psi(x)
%
% The cost matrices Q and R are tuned to penalize deviations in the physical
% states (cart position and pendulum angle) and control effort.
Q = diag([100, 1, 100, 1, 50, 50, 10, 10, 10, 1]);
R = 1;
[K_koop,~,~] = dlqr(A_koop, B_koop, Q, R);

%% 5. Closed-Loop Simulation Using ode15s with Control Saturation
% Here we simulate the original nonlinear dynamics under the Koopman-based LQR.
% The controller computes:
%      u = -K_koop * psi(x)
% and we apply a saturation to keep u within ±100.
%
% Use ode15s (a stiff solver) with relaxed tolerances.
x0 = [0; 0; 0.1; 0];  % initial condition: cart at 0, pendulum angle 0.1 rad, zero velocities
Tsim_cl = 10;  % total simulation time in seconds
options = odeset('RelTol',1e-6, 'AbsTol',1e-8);

[t, x_cl] = ode15s(@(t,x) closedLoopDynamics(t, x, a, b, c, m_p, g, l, K_koop), [0 Tsim_cl], x0, options);

%% 6. Plot the Closed-Loop Results
figure;
subplot(2,1,1);
plot(t, x_cl(:,1), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Cart Position x (m)');
title('Cart Position under Koopman-based LQR');

subplot(2,1,2);
plot(t, x_cl(:,3), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Pendulum Angle \theta (rad)');
title('Pendulum Angle under Koopman-based LQR');

figure;
% Compute and plot the control input over time for visualization
u_cl = zeros(length(t),1);
for i = 1:length(t)
    u_val = -K_koop * liftState(x_cl(i,:)');
    % Apply saturation for clarity
    max_u = 100;
    u_cl(i) = max(min(u_val, max_u), -max_u);
end
plot(t, u_cl, 'LineWidth',2);
xlabel('Time (s)'); ylabel('Control Input F (N)');
title('Control Input');

%% ----------------- Local Functions ------------------------

function xdot = nonlinearDynamics(x, F, a, b, c, m_p, g, l)
    % nonlinearDynamics computes the state derivative for the two-wheeled inverted pendulum.
    % State: x = [x; x_dot; theta; theta_dot]
    % F: control force.
    %
    % Equations of motion:
    %   x_dot      = x(2)
    %   x_ddot     = [ c*(F + b*sin(theta)*theta_dot^2) - b*cos(theta)*m_p*g*l*sin(theta) ] / ( a*c - b^2*cos(theta)^2 )
    %   theta_dot  = x(4)
    %   theta_ddot = [ a*(m_p*g*l*sin(theta)) - b*cos(theta)*(F + b*sin(theta)*theta_dot^2) ] / ( a*c - b^2*cos(theta)^2 )
    
    theta = x(3);
    theta_dot = x(4);
    
    % Denomintor (theta-dependent)
    Delta = a*c - b^2*cos(theta)^2;
    
    % Compute accelerations
    x_ddot = ( c*(F + b*sin(theta)*theta_dot^2) - b*cos(theta)*m_p*g*l*sin(theta) ) / Delta;
    theta_ddot = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*(F + b*sin(theta)*theta_dot^2) ) / Delta;
    
    xdot = [ x(2); x_ddot; theta_dot; theta_ddot ];
end

function psi = liftState(x)
    % liftState defines the dictionary of observables (the lifted state) for x.
    % The dictionary used here is:
    %   psi(x) = [ x; x_dot; theta; theta_dot; sin(theta); cos(theta); x^2; theta^2; x*theta; 1 ]
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
    % The control is computed using the lifted state and the Koopman LQR gain.
    z = liftState(x);
    u = -K_koop * z;
    % Apply saturation to prevent excessively high control values.
    max_u = 100;
    u = max(min(u, max_u), -max_u);
    dxdt = nonlinearDynamics(x, u, a, b, c, m_p, g, l);
end
