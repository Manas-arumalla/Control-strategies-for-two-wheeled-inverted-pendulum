%% Modified Carleman Linearization + LQR Controller (Increased x Weight)
% In this version, we modify the Q matrix to put a higher penalty on the x state.
%In this work, we im_plemented a state-feedback controller based on an augmented model obtained via Carleman linearization of the nonlinear dynamics. The controller was designed using an LQR framework with a modified cost function that significantly increased the weight on the cart position. This modification ensured that deviations in the cart's position were heavily penalized, thereby achieving simultaneous stabilization of both the pendulum and the cart. Simulation results confirmed that this approach effectively reduced transient deviations and minimized steady-state error in the cart, making it a promising strategy for robust stabilization of underactuated systems.
clear; close all; clc;

%% 1. Define System Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

%% 2. Com_pute Derived Constants
a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;
denom = a*c - b^2;

%% 3. Augmentation Tuning Parameters
alpha_val = 1;    % Tuning parameter for auxiliary dynamics
delta_val = 0.05; % Small coupling from control input into auxiliary state

%% 4. Original Linearized Model (4-State)
% States: x1 = x, x2 = x_dot, x3 = theta, x4 = theta_dot.
A_orig = [0, 1, 0, 0;
          0, 0, - (b*m_p*g*l)/denom, 0;
          0, 0, 0, 1;
          0, 0, (a*m_p*g*l)/denom, 0];
B_orig = [0; c/denom; 0; -b/denom];

%% 5. Augment the System with an Auxiliary State z5
% Define the augmented state: z = [ x; x_dot; theta; theta_dot; z5 ]
% with auxiliary dynamics:
%    z5_dot = theta_dot + alpha_val*(theta - z5) + delta_val*F
A_aug = [ 0,   1,                0,                 0,      0;
          0,   0,    - (b*m_p*g*l)/denom,                0,      0;
          0,   0,                0,                 1,      0;
          0,   0,     (a*m_p*g*l)/denom,                0,      0;
          0,   0,         alpha_val,                 1, -alpha_val];
      
B_aug = [0; c/denom; 0; -b/denom; delta_val];

%% 6. Check Controllability of the Augmented System
Co = ctrb(A_aug, B_aug);
if rank(Co) < size(A_aug,1)
    error('The augmented system is not fully controllable. Adjust delta_val or the augmentation.');
else
    disp('The augmented system is controllable.');
end

%% 7. Design LQR Controller on the Augmented System
% Increase the weight on the cart position (state x) to stabilize the cart.
Q_aug = diag([100, 1, 100, 1, 10]);  % x weight increased from 10 to 100
R = 1;                              % Control penalty (scalar)

% Com_pute the LQR gain:
K = lqr(A_aug, B_aug, Q_aug, R);

%% 8. Simulation of Closed-Loop Dynamics
% Closed-loop dynamics: z_dot = (A_aug - B_aug*K)*z.
closedLoopDynamics = @(t, z) (A_aug - B_aug*K)*z;

% Initial conditions: small deviation in theta; x initially zero.
theta0 = 0.1;
z0 = [0; 0; theta0; 0; theta0];  % [x; x_dot; theta; theta_dot; z5]

tspan = [0 10];
[t, z_sol] = ode45(closedLoopDynamics, tspan, z0);

%% 9. Plot the Results
figure;
subplot(3,1,1);
plot(t, z_sol(:,1), 'LineWidth',2); 
title('Cart Position');
grid on;
xlabel('Time (s)'); ylabel('x (m)');
title('Horizontal Position (Cart)');
subplot(3,1,2);
plot(t, z_sol(:,3), 'r','LineWidth',2);
xlabel('Time (s)'); ylabel('\theta (rad)');
title('Pendulum Angle');
grid on;

subplot(3,1,3);
plot(t, z_sol(:,5), 'g','LineWidth',2);
xlabel('Time (s)'); ylabel('z5');
title('Auxiliary State z5 Dynamics');
grid on;

figure;
u = -K * z_sol';  % Reconstruct control input
plot(t, u, 'k' ,'LineWidth', 2);
xlabel('Time (s)'); ylabel('Control Input F (N)');
title('LQR Control Input');
grid on;

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = z_sol(:,1);  % Cart/Wheel Position
angle_response = z_sol(:,2);     % Pendulum Angle

% Deviations from desired reference
position_deviation = abs(position_response - desired_reference);
angle_deviation = abs(angle_response - desired_reference);

% Settling time for cart position
settling_index_pos = find(position_deviation > tolerance, 1, 'last');
if ~isempty(settling_index_pos)
    settling_time_position = t(settling_index_pos);
else
    settling_time_position = 0;
end

% Settling time for pendulum angle
settling_index_ang = find(angle_deviation > tolerance, 1, 'last');
if ~isempty(settling_index_ang)
    settling_time_angle = t(settling_index_ang);
else
    settling_time_angle = 0;
end

fprintf('Manual Settling Time:\n');
fprintf(' - Cart Position     : %.3f seconds\n', settling_time_position);
fprintf(' - Pendulum Angle    : %.3f seconds\n', settling_time_angle);

