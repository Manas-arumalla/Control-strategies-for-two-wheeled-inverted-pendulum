%% Revised Segway Model with Improved Sontag's Feedback Control
clear; close all; clc;

%% System Parameters (example values; adjust as needed)
m_w = 1.0;      % mass of each wheel [kg]
I_W = 0.02;     % moment of inertia of each wheel about its center [kg*m^2]
m_p = 5.0;      % mass of pendulum [kg]
I_p = 0.2;      % moment of inertia of pendulum about its center [kg*m^2]
l   = 0.5;      % distance from axle to pendulum center of mass [m]
r   = 0.1;      % wheel radius [m]
g   = 9.81;     % gravitational acceleration [m/s^2]

% Derived constants
a = 2*m_w + m_p + 2*I_W/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;

%% Simulation Setup
tspan = [0 10];
% Initial state: [x; xdot; theta; theta_dot]
x0 = [0; 0; 0.1; 0];  % small perturbation from the desired equilibrium (all zeros)

%% Controller Tuning Parameters
% Weights for the candidate Lyapunov function:
% V = 0.5*(w1*x^2 + w2*xdot^2 + w3*theta^2 + w4*theta_dot^2) + w5*x*theta
% The cross term (w5*x*theta) couples the horizontal and angular states.
w1 = 1.0;  % weight for x (position)
w2 = 1.0;  % weight for xdot (velocity)
w3 = 5.0;  % weight for theta (pendulum angle)
w4 = 1.0;  % weight for theta_dot (angular velocity)
w5 = 2.0;  % cross-coupling weight (choose such that w1*w3 > w5^2 to keep V positive definite)

% Control saturation limit (to prevent unrealistic force values)
F_max = 50;

%% Run Simulation Using ode45
[t, X] = ode45(@(t,x) segwayDynamics(t, x, a, b, c, m_p, g, l, w1, w2, w3, w4, w5, F_max), tspan, x0);

%% Plot Results
figure;
subplot(2,1,1);
plot(t, X(:,3), 'LineWidth',2);
xlabel('Time [s]'); ylabel('\theta [rad]');
title('Pendulum Angle');
grid on;

subplot(2,1,2);
plot(t, X(:,1), 'LineWidth',2);
xlabel('Time [s]'); ylabel('x [m]');
title('Axle Horizontal Displacement');
grid on;

%% Dynamics Function with Modified CLF and Sontag's Formula
function dx = segwayDynamics(~, x, a, b, c, m_p, g, l, w1, w2, w3, w4, w5, F_max)
    % State vector: x = [x; xdot; theta; theta_dot]
    x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4);
    
    % Pre-calculate trigonometric terms and Delta
    cosTheta = cos(x3);
    sinTheta = sin(x3);
    Delta = a*c - b^2*cosTheta^2;
    
    % Compute drift terms (dynamics without control force F)
    f2 = (c*b*sinTheta*x4^2 - b*m_p*g*l*sinTheta*cosTheta) / Delta;
    f4 = (-b^2*sinTheta*cosTheta*x4^2 + a*m_p*g*l*sinTheta) / Delta;
    
    % Control input appears in the affine terms:
    g2 = c / Delta;
    g4 = -b*cosTheta / Delta;
    
    % Assemble the drift vector and control influence vector
    f = [ x2;
          f2;
          x4;
          f4 ];
    g_vec = [ 0;
              g2;
              0;
              g4 ];
          
    % --- Modified Candidate Lyapunov Function ---
    % V = 0.5*(w1*x1^2 + w2*x2^2 + w3*x3^2 + w4*x4^2) + w5*x1*x3
    % Its gradient is:
    dVdx = [ w1*x1 + w5*x3;
             w2*x2;
             w3*x3 + w5*x1;
             w4*x4 ];
    
    % Compute Lie derivatives along f and g_vec:
    LfV = dVdx' * f;
    LgV = dVdx' * g_vec;
    
    % --- Sontag's Universal Formula ---
    % Regularization parameter to avoid division by zero:
    eps = 1e-4;
    if abs(LgV) > eps
        F_unsat = - ( LfV + sqrt(LfV^2 + LgV^4 ) ) / LgV;
    else
        F_unsat = 0;
    end
    
    % Saturate the control input to ensure bounded actuation:
    F = max(min(F_unsat, F_max), -F_max);
    
    % Full state dynamics including control:
    dx = f + g_vec * F;
end
