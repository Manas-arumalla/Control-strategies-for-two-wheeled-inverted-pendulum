%%
% Hybrid Adaptive Sliding Mode Control with ADRC and MRAC
% Overview:
% The proposed control strategy is designed to robustly stabilize both the wheels (cart position) and the pendulum (angle) of a two-wheeled inverted pendulum system. The architecture combines Active Disturbance Rejection Control (ADRC), Sliding Mode Control (SMC), and Model Reference Adaptive Control (MRAC) into a single unified scheme. This approach is intended to handle system uncertainties and external disturbances while maintaining low computational complexity.
% 
% Extended State Observers (ESOs):
% Two ESOs are implemented—one for the cart (wheels) and one for the pendulum. Each ESO estimates not only the system states (cart position/velocity and pendulum angle/angular velocity) but also the lumped disturbances acting on them. The observer gains (
% 𝐿
% 1
% L1, 
% 𝐿
% 2
% L2, 
% 𝐿
% 3
% L3) are determined based on a chosen observer bandwidth, ensuring rapid convergence of estimation errors. These disturbance estimates are critical as they allow the controller to cancel out the effects of unmodeled dynamics and external perturbations.
% 
% Sliding Mode Control (SMC):
% Instead of using a basic PD law, a sliding mode controller is adopted to provide robust nonlinear feedback. For each subsystem, sliding surfaces are defined as:
% 
% 𝑠
% 𝑥
% =
% 𝑒
% 𝑥
% +
% 𝜆
% 𝑥
% 𝑒
% ˙
% 𝑥
% s 
% x
% ​
%  =e 
% x
% ​
%  +λ 
% x
% ​
%   
% e
% ˙
%   
% x
% ​
%   for the cart (where 
% 𝑒
% 𝑥
% e 
% x
% ​
%   is the cart position error), and
% 𝑠
% 𝜃
% =
% 𝑒
% 𝜃
% +
% 𝜆
% 𝜃
% 𝑒
% ˙
% 𝜃
% s 
% θ
% ​
%  =e 
% θ
% ​
%  +λ 
% θ
% ​
%   
% e
% ˙
%   
% θ
% ​
%   for the pendulum (where 
% 𝑒
% 𝜃
% e 
% θ
% ​
%   is the pendulum angle error).
% A weighted combination of these surfaces, 
% 𝑠
% =
% 𝛼
% 𝑠
% 𝑥
% +
% (
% 1
% −
% 𝛼
% )
% 𝑠
% 𝜃
% s=αs 
% x
% ​
%  +(1−α)s 
% θ
% ​
%  , is used as the error signal. The control law consists of a nominal part 
% 𝑢
% 𝑛
% 𝑜
% 𝑚
% =
% −
% 𝐾
% 𝑠
% u 
% nom
% ​
%  =−Ks (with 
% 𝐾
% K being an adaptive gain) and a robust term 
% 𝑢
% 𝑟
% 𝑜
% 𝑏
% 𝑢
% 𝑠
% 𝑡
% =
% −
% 𝜂
% tanh
% ⁡
% (
% 𝑠
% /
% 𝜖
% )
% u 
% robust
% ​
%  =−ηtanh(s/ϵ) to counteract uncertainties while reducing chattering. The overall control signal is adjusted by canceling the disturbance estimates from both ESOs.
% 
% Adaptive Gain Tuning via MRAC:
% To enhance performance under varying conditions, the controller uses an MRAC-inspired adaptation law to update the sliding mode gain 
% 𝐾
% K online. This adaptive mechanism increases 
% 𝐾
% K in proportion to the square of the sliding surface 
% 𝑠
% s, thereby compensating for model uncertainties and ensuring robust error convergence.
% 
% Implementation of Full Nonlinear Dynamics:
% The computed control input is applied to the full nonlinear model of the two-wheeled inverted pendulum. This model captures the coupling between the translational dynamics of the cart and the rotational dynamics of the pendulum. An Euler integration method is used for state updates, and control saturation is enforced to maintain practical actuator limits.
% 
% Summary:
% 
% ESOs provide real-time estimates of both states and disturbances, enabling effective cancellation of unmodeled dynamics.
% SMC replaces a basic PD controller with a robust nonlinear feedback law that drives the sliding surfaces to zero, ensuring rapid error convergence.
% MRAC-based adaptation continuously tunes the controller gains, making the system resilient to uncertainties and disturbances.
% Together, these components stabilize both the cart's position and the pendulum's angle with high robustness and reduced computational complexity.
%%
clc; clear; close all;

%% System Parameters
m_w = 0.432;           % Mass of each wheel [kg]
m_p = 5.0;             % Mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2; % Inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;       % Inertia of the pendulum [kg*m^2]
l = 0.4;               % Distance from axle to pendulum COM [m]
r = 0.0726;            % Wheel radius [m]
g = 9.81;              % Gravity [m/s^2]

% Derived coefficients from the dynamic model:
a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;

% Nominal gain factor (for disturbance cancellation)
% (computed at equilibrium; you may adjust this as needed)
b0 = -b / Delta;  

%% ADRC-ESO Gains for Cart (x) and Pendulum (theta)
w0 = 30;         % Observer bandwidth
L1 = 3*w0;       % ESO gains (same for both subsystems for simplicity)
L2 = 3*w0^2;
L3 = w0^3;

%% Sliding Mode Control (SMC) and Adaptive Gain Parameters
% Sliding surface parameters for cart and pendulum:
lambda_x = 5;          % SMC slope for cart
lambda_theta = 10;     % SMC slope for pendulum
% Weighting factor to combine sliding surfaces (0<=alpha<=1)
alpha_weight = 0.5;    % equal weight for cart and pendulum

% Nominal sliding mode control gain (will be adapted online)
K = 50;                % Initial adaptive gain
eta = 5;               % Discontinuous (robust) gain
epsilon = 0.01;        % Smoothing parameter for tanh()

% Adaptation rate for gain K (MRAC law)
gamma = 10;

%% Simulation Parameters
dt = 0.001;
T_final = 10;
t = 0:dt:T_final;
N = length(t);

% State vector: [cart position; cart velocity; pendulum angle; pendulum angular velocity]
x = zeros(4, N);
% Set initial conditions: small pendulum deviation, cart initially at rest
x(:,1) = [0; 0; 0.1; 0];

% Preallocate control and gain histories
u_total = zeros(1, N);
K_hist = zeros(1, N);
K_hist(1) = K;

%% ESO Initialization for Cart and Pendulum
% For Cart:
z_x = x(1,1);       % estimated cart position
z_xdot = x(2,1);    % estimated cart velocity
z_dist_x = 0;       % estimated disturbance for cart

% For Pendulum:
z_theta = x(3,1);       % estimated pendulum angle
z_thetadot = x(4,1);    % estimated pendulum angular velocity
z_dist_theta = 0;       % estimated disturbance for pendulum

%% Main Simulation Loop
for k = 1:N-1
    %% Measurements from plant (available states)
    x_meas = x(:,k);
    % For convenience:
    cart_pos = x_meas(1);
    cart_vel = x_meas(2);
    theta = x_meas(3);
    theta_dot = x_meas(4);
    
    %% --- Extended State Observer (ESO) Updates ---
    % ESO for Cart:
    e_x_obs = cart_pos - z_x;
    z_x = z_x + dt*(z_xdot + L1*e_x_obs);
    z_xdot = z_xdot + dt*(z_dist_x + L2*e_x_obs);
    z_dist_x = z_dist_x + dt*(L3*e_x_obs);
    
    % ESO for Pendulum:
    e_theta_obs = theta - z_theta;
    z_theta = z_theta + dt*(z_thetadot + L1*e_theta_obs);
    z_thetadot = z_thetadot + dt*(z_dist_theta + L2*e_theta_obs);
    z_dist_theta = z_dist_theta + dt*(L3*e_theta_obs);
    
    %% --- Define Sliding Surfaces ---
    % Define errors (desired values are zero for stabilization)
    e_x = 0 - z_x;               % cart position error
    e_xdot = 0 - z_xdot;         % cart velocity error
    e_theta = 0 - z_theta;       % pendulum angle error
    e_thetadot = 0 - z_thetadot; % pendulum angular velocity error
    
    % Sliding surfaces for cart and pendulum:
    s_x = e_x + lambda_x * e_xdot;
    s_theta = e_theta + lambda_theta * e_thetadot;
    
    % Combined sliding surface (weighted sum)
    s = alpha_weight * s_x + (1 - alpha_weight) * s_theta;
    
    %% --- Adaptive Sliding Mode Control Law ---
    % Nominal control part using the sliding variable with adaptive gain K
    u_nom = - K * s;
    % Robust/discontinuous term (smooth approximation)
    u_robust = - eta * tanh(s/epsilon);
    
    % Combined nominal control before disturbance cancellation
    u_ASMC = u_nom + u_robust;
    
    % Cancel estimated disturbances (sum of both ESOs' disturbance estimates)
    u = (u_ASMC - (z_dist_x + z_dist_theta)) / b0;
    
    % Saturate control input to avoid unrealistic commands
    u_sat = 10;
    u = max(min(u, u_sat), -u_sat);
    u_total(k) = u;
    
    %% --- Adaptive Gain Update (MRAC Law) ---
    % A simple adaptation law: increase gain proportional to sliding surface magnitude
    K = K + gamma * s^2 * dt;
    % Optionally, you can include a forgetting factor or bounds on K here
    K_hist(k+1) = K;
    
    %% --- Plant Dynamics Simulation (Full Nonlinear Model) ---
    % Dynamics as derived:
    % a*ddot{x} + b*cos(theta)*ddot{theta} - b*sin(theta)*(theta_dot)^2 = u
    % b*cos(theta)*ddot{x} + c*ddot{theta} - m_p*g*l*sin(theta) = 0
    % Solve for ddot{x} and ddot{theta}:
    denom = a*c - b^2 * cos(theta)^2;
    ddx = ( c*( u + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*( m_p*g*l*sin(theta) ) ) / denom;
    ddtheta = ( a*( m_p*g*l*sin(theta) ) - b*cos(theta)*( u + b*sin(theta)*theta_dot^2 ) ) / denom;
    
    % State derivative
    dx = [ cart_vel;
           ddx;
           theta_dot;
           ddtheta ];
    % Euler integration update
    x(:,k+1) = x(:,k) + dt * dx;
end

%% Plot Results
figure;
subplot(3,1,1);
plot(t, x(1,:), 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('Cart Position Stabilization');
grid on;

subplot(3,1,2);
plot(t, x(3,:), 'LineWidth', 1.5);
ylabel('Pendulum Angle (rad)');
title('Pendulum Angle Stabilization');
grid on;

subplot(3,1,3);
plot(t(1:end-1), u_total(1:end-1), 'LineWidth', 1.5);
ylabel('Control Input (N)');
xlabel('Time (s)');
title('Control Effort');
grid on;

figure;
plot(t, K_hist, 'LineWidth', 1.5);
ylabel('Adaptive Gain K');
xlabel('Time (s)');
title('Evolution of Adaptive Gain');
grid on;
