%% optimize_and_plot.m
% This script uses a genetic algorithm (GA) to tune the IDA-PBC controller
% gains for a Segway-like system with the goal of minimizing the settling time.
% After obtaining the optimal parameters, it simulates the closed-loop system
% and plots the cart position and pendulum angle.

clear; clc; close all;

%% --- GA Setup ---
% Decision variables: [k_x, k_dx, k_theta, k_theta_d]
% Lower and upper bounds for the gains:
lb = [2, 2, 2, 2];
ub = [50, 50, 100, 50];

% GA options (tune PopulationSize, MaxGenerations, etc. as needed)
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50, 'MaxGenerations', 50);

% Number of decision variables
nvars = 4;

% Run the genetic algorithm to minimize the objective function (settling time)
[optParams, fval] = ga(@objFun, nvars, [], [], [], [], lb, ub, [], options);

fprintf('Optimized parameters:\n');
fprintf('  k_x       = %.3f\n', optParams(1));
fprintf('  k_dx      = %.3f\n', optParams(2));
fprintf('  k_theta   = %.3f\n', optParams(3));
fprintf('  k_theta_d = %.3f\n', optParams(4));
fprintf('Optimal objective value (settling time): %.3f s\n', fval);

%% --- Simulate the System with the Optimal Gains ---
% System parameters
m_w = 0.432;                  % mass of each wheel [kg]
r   = 0.0726;                 % wheel radius [m]
I_W = 0.5 * m_w * r^2;        % moment of inertia of each wheel [kg*m^2]

m_p = 5.0;                    % mass of the pendulum [kg]
l   = 0.4;                    % distance from axle to pendulum's COM [m]
I_p = 5 * (0.4)^2;            % pendulum moment of inertia [kg*m^2]

g   = 9.81;                   % gravitational acceleration [m/s^2]

% Derived parameters
a = 2*m_w + m_p + 2*I_W/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;

% Simulation settings
tspan = [0 5];                % simulation time [s]
x0 = [0; 0; 0.1; 0];           % initial state: [x; x_dot; theta; theta_dot]

% Simulate using ode45 with optimal gains
[t, X] = ode45(@(t, X) segwayDynamics(t, X, a, b, c, m_p, g, l, ...
    optParams(1), optParams(2), optParams(3), optParams(4)), tspan, x0);

%% --- Plotting the Results ---
figure;
subplot(2,1,1);
plot(t, X(:,1), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Cart Position, x (m)');
title('Cart Position with Optimized Gains');
grid on;

subplot(2,1,2);
plot(t, X(:,3), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Pendulum Angle, \theta (rad)');
title('Pendulum Angle with Optimized Gains');
grid on;

%% --- Nested Functions ---
% Objective function for GA: computes settling time (in seconds)
function J = objFun(params)
    % Unpack controller gains
    k_x       = params(1);
    k_dx      = params(2);
    k_theta   = params(3);
    k_theta_d = params(4);
    
    % System parameters (must match those used in simulation)
    m_w = 0.432;
    r   = 0.0726;
    I_W = 0.5 * m_w * r^2;
    
    m_p = 5.0;
    l   = 0.4;
    I_p = 5 * (0.4)^2;
    g   = 9.81;
    
    % Derived parameters
    a = 2*m_w + m_p + 2*I_W/(r^2);
    b = m_p * l;
    c = m_p * l^2 + I_p;
    
    % Simulation settings
    tspan = [0 5];
    x0 = [0; 0; 0.1; 0];
    
    % Run simulation with the current gains
    [t, X] = ode45(@(t, X) segwayDynamics(t, X, a, b, c, m_p, g, l, k_x, k_dx, k_theta, k_theta_d), tspan, x0);
    
    % Calculate settling times for x and theta based on a tolerance (e.g., 5% threshold)
    errBound = 0.05;
    settleTime_x     = settlingTime(t, X(:,1), errBound);
    settleTime_theta = settlingTime(t, X(:,3), errBound);
    
    % The objective is the maximum of the two settling times
    J = max(settleTime_x, settleTime_theta);
    
    % Penalize if settling did not occur within simulation time
    if isnan(J)
        J = 1e6;
    end
end

% Function to compute settling time for a given signal y(t)
function T = settlingTime(t, y, tol)
    idx = find(abs(y) < tol, 1);
    if isempty(idx)
        T = NaN;
    else
        T = t(idx);
    end
end

% Dynamics of the Segway-like system with the augmented IDA-PBC controller
function dXdt = segwayDynamics(~, X, a, b, c, m_p, g, l, k_x, k_dx, k_theta, k_theta_d)
    % State vector: X = [x; x_dot; theta; theta_dot]
    x       = X(1);
    x_dot   = X(2);
    theta   = X(3);
    theta_dot = X(4);
    
    % Augmented IDA-PBC Control Law:
    % F = - (a - (b^2*cos(theta)^2)/c) * (k_dx*x_dot + k_x*x)
    %     + b*sin(theta)*theta_dot^2
    %     - (b*cos(theta)/c) * (m_p*g*l*sin(theta) - k_theta*theta - k_theta_d*theta_dot)
    F = - (a - (b^2*cos(theta)^2)/c) * (k_dx*x_dot + k_x*x) ...
        + b*sin(theta)*theta_dot^2 ...
        - (b*cos(theta)/c) * (m_p*g*l*sin(theta) - k_theta*theta - k_theta_d*theta_dot);
    
    % The system dynamics are modeled as:
    % [ a          b*cos(theta) ] [ x_ddot     ] = [ F + b*sin(theta)*theta_dot^2 ]
    % [ b*cos(theta)      c      ] [ theta_ddot ]   [ m_p*g*l*sin(theta)           ]
    M = [ a, b*cos(theta); 
          b*cos(theta), c ];
    
    RHS = [ F + b*sin(theta)*theta_dot^2;
            m_p*g*l*sin(theta) ];
    
    % Solve for the accelerations [x_ddot; theta_ddot]
    accel = M \ RHS;
    
    dXdt = [x_dot; accel(1); theta_dot; accel(2)];
end
