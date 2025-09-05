%% Modified IDA-PBC Controller Simulation for a Segway-like System
% This script simulates the closed-loop dynamics of a two-wheeled inverted
% pendulum (Segway-like system) using an augmented IDA-PBC control law with an
% extra damping term for the pendulum angle.

clear; clc; close all;

%% System Parameters
% Wheel parameters
m_w = 0.432;                  % mass of each wheel [kg]
r   = 0.0726;                 % wheel radius [m]
I_W = 0.5 * m_w * r^2;          % moment of inertia of each wheel [kg*m^2]

% Pendulum parameters
m_p = 5.0;                    % mass of the pendulum [kg]
l   = 0.4;                    % distance from axle to pendulum's center of mass [m]
I_p = 5 * (0.4)^2;            % moment of inertia of the pendulum [kg*m^2]

g   = 9.81;                   % gravitational acceleration [m/s^2]

%% Derived Parameters
% Effective horizontal inertia, coupling, and pendulum inertia
a = 2*m_w + m_p + 2*I_W/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;

%% Control Gains
% Gains for shaping the horizontal dynamics (tune these as needed)
k_x  = 10;       % proportional gain for horizontal position
k_dx = 5;        % derivative (damping) gain for horizontal velocity

% Gains for shaping the pendulum dynamics:
k_theta   = 50;    % gain to shape the pendulum potential (helps make theta=0 stable)
k_theta_d = 10;    % additional damping gain for the pendulum angle

%% Simulation Settings
tspan = [0 5];    % simulation time [s]
% Initial state: [x; x_dot; theta; theta_dot]
% (x: horizontal displacement, theta: pendulum angle in radians)
x0 = [0; 0; 0.1; 0];  

%% ODE Solver
[t, X] = ode45(@(t, X) segwayDynamics(t, X, a, b, c, m_p, g, l, k_x, k_dx, k_theta, k_theta_d), tspan, x0);

%% Plotting the Results
figure;
subplot(2,1,1);
plot(t, X(:,1), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('x (m)');
title('Cart Position');
grid on;

subplot(2,1,2);
plot(t, X(:,3), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('\theta (rad)');
title('Pendulum Angle');
grid on;

%% Function: segwayDynamics
function dXdt = segwayDynamics(~, X, a, b, c, m_p, g, l, k_x, k_dx, k_theta, k_theta_d)
    % The state vector X = [x; x_dot; theta; theta_dot]
    x       = X(1);
    x_dot   = X(2);
    theta   = X(3);
    theta_dot = X(4);
    
    % Modified IDA-PBC Control Law with extra pendulum damping:
    %
    % The original control law (without extra theta damping) was:
    %
    %   F = - (a - (b^2*cos(theta)^2)/c) * (k_dx*x_dot + k_x*x)
    %       + b*sin(theta)*theta_dot^2 - (b*cos(theta)/c) * (m_p*g*l*sin(theta) - k_theta*theta)
    %
    % To improve pendulum stabilization we add a damping term proportional to theta_dot:
    %
    %   F = - (a - (b^2*cos(theta)^2)/c) * (k_dx*x_dot + k_x*x)
    %       + b*sin(theta)*theta_dot^2 - (b*cos(theta)/c) * (m_p*g*l*sin(theta) - k_theta*theta - k_theta_d*theta_dot)
    %
    F = - (a - (b^2*cos(theta)^2)/c) * (k_dx*x_dot + k_x*x) ...
        + b*sin(theta)*theta_dot^2 ...
        - (b*cos(theta)/c) * (m_p*g*l*sin(theta) - k_theta*theta - k_theta_d*theta_dot);
    
    % System dynamics written in matrix form:
    %
    %   [ a          b*cos(theta) ] [ x_ddot     ] = [ F + b*sin(theta)*theta_dot^2 ]
    %   [ b*cos(theta)      c      ] [ theta_ddot ]   [ m_p*g*l*sin(theta)           ]
    %
    M = [ a, b*cos(theta); 
          b*cos(theta), c ];
    
    RHS = [ F + b*sin(theta)*theta_dot^2;
            m_p*g*l*sin(theta) ];
    
    % Solve for the accelerations: [x_ddot; theta_ddot]
    accel = M\RHS;
    
    % Construct the derivative of the state vector
    dXdt = zeros(4,1);
    dXdt(1) = x_dot;
    dXdt(2) = accel(1);
    dXdt(3) = theta_dot;
    dXdt(4) = accel(2);
end
