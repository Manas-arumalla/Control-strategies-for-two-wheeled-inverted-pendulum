%% EPSAC Implementation for Stabilizing Both Cart Position and Pendulum Angle
clear; clc; close all;

%% Define Plant Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % Length of the pendulum [m]
r   = 0.0726;     % wheel radius [m]
g   = 9.81;       % gravitational acceleration [m/s^2]

% Derived parameters from the linearized model:
%  a*x_ddot + b*theta_ddot = F
%  b*x_ddot + c*theta_ddot = d*theta
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

Delta = a*c - b^2;  % determinant

%% Construct the Linearized State–Space Model
% States: x = [ cart_position; cart_velocity; pendulum_angle; pendulum_angular_velocity ]
A = [ 0,     1,                      0,              0;
      0,     0,               -b*d/Delta,              0;
      0,     0,                      0,              1;
      0,     0,                a*d/Delta,              0];

B = [ 0;
      c/Delta;
      0;
     -b/Delta];

% Outputs: both cart position and pendulum angle
C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

sys_ss = ss(A, B, C, D);

%% Discretize the System
Ts = 0.1;  % Sampling time (seconds)
sys_d = c2d(sys_ss, Ts);

%% EPSAC Design Parameters
Np = 25;         % Prediction Horizon (number of steps)
Nc = 6;          % Control Horizon (number of optimized control moves)
lambda = 0.001;    % Weight on control effort in cost function

% Input constraints:
u_min = -10;
u_max = 10;

% Initialize the estimated model as the discretized (nominal) model.
A_est = sys_d.A;
B_est = sys_d.B;

%% Simulation Setup
T_sim = 10;                    % Total simulation time (seconds)
N_sim = T_sim / Ts;            % Number of simulation steps

% Preallocate storage vectors:
x = zeros(4, N_sim+1);         % State trajectory
x(:,1) = [0; 0; 0.1; 0];     % Small initial deviation for demonstration
u_store = zeros(1, N_sim);     % Control inputs
y_store = zeros(2, N_sim);     % Outputs (cart position and pendulum angle)
time = (0:N_sim-1)*Ts;         % Time vector

% Define a constant reference trajectory for stabilization (zero for both outputs).
ref = [0; 0];

%% Main EPSAC Loop
for k = 1:N_sim
    % Current state measurement:
    xk = x(:,k);
    
    % Construct the reference trajectory over the prediction horizon.
    R = repmat(ref, 1, Np);
    
    % Optimization: Decide future control moves.
    % We optimize Nc control moves. Beyond Nc, the control action is held constant.
    U0 = zeros(Nc,1);  % initial guess
    options = optimoptions('fmincon','Display','off','Algorithm','sqp');
    
    % Solve the optimization problem using fmincon.
    [U_opt, ~] = fmincon(@(U) costFunction(U, xk, R, A_est, B_est, Nc, Np, C, lambda), ...
                          U0, [], [], [], [], ...
                          u_min*ones(Nc,1), u_max*ones(Nc,1), [], options);
    
    % Apply the first control input from the optimal sequence.
    u_k = U_opt(1);
    u_store(k) = u_k;
    
    % Simulate the actual plant using the discretized model.
    x(:,k+1) = sys_d.A * xk + sys_d.B * u_k;
    
    % Store the measured output.
    y_store(:,k) = C * xk;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Adaptive Update (Simple Parameter Adaptation)
    % Adjust A_est and B_est using the prediction error.
    adaptation_gain = 0.001;
    pred_x = A_est * xk + B_est * u_k;   % Predicted next state using the estimated model.
    error_x = x(:,k+1) - pred_x;          % Prediction error.
    
    % Update A_est and B_est based on current state and control input.
    A_est = A_est + adaptation_gain * error_x * xk';
    B_est = B_est + adaptation_gain * error_x * u_k;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%% Plot the Results
figure;
subplot(2,1,1);
plot(time, x(1,1:end-1),'b','LineWidth',1.5);
xlabel('Time (s)');
ylabel('Cart Position');
title('Cart Position vs. Time (Stabilization)');
grid on;

subplot(2,1,2);
plot(time, x(3,1:end-1),'r','LineWidth',1.5);
xlabel('Time (s)');
ylabel('Pendulum Angle (rad)');
title('Pendulum Angle vs. Time (Stabilization)');
grid on;

figure;
plot(time, u_store,'k','LineWidth',1.5);
xlabel('Time (s)');
ylabel('Control Input');
title('Control Input vs. Time');
grid on;

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = x(1,1:end-1);  % Cart/Wheel Position
angle_response = x(3,1:end-1);     % Pendulum Angle

% Deviations from desired reference
position_deviation = abs(position_response - desired_reference);
angle_deviation = abs(angle_response - desired_reference);

% Settling time for cart position
settling_index_pos = find(position_deviation > tolerance, 1, 'last');
if ~isempty(settling_index_pos)
    settling_time_position = time(settling_index_pos);
else
    settling_time_position = 0;
end

% Settling time for pendulum angle
settling_index_ang = find(angle_deviation > tolerance, 1, 'last');
if ~isempty(settling_index_ang)
    settling_time_angle = time(settling_index_ang);
else
    settling_time_angle = 0;
end

fprintf('Manual Settling Time:\n');
fprintf(' - Cart Position     : %.3f seconds\n', settling_time_position);
fprintf(' - Pendulum Angle    : %.3f seconds\n', settling_time_angle);


%% Cost Function for EPSAC Optimization
% Computes the cost over the prediction horizon given a candidate control sequence U.
function J = costFunction(U, x0, R, A_est, B_est, Nc, Np, C, lambda)
    % Extend the control sequence for the full prediction horizon.
    U_extended = [U; repmat(U(end), Np - Nc, 1)];
    x_pred = x0;
    J = 0;
    % Simulate the predicted output trajectory over Np steps.
    for i = 1:Np
        % Update the predicted state.
        x_pred = A_est * x_pred + B_est * U_extended(i);
        y_pred = C * x_pred;
        % Calculate the tracking error.
        e = R(:,i) - y_pred;
        % Accumulate cost: quadratic error + control effort penalty.
        J = J + (e' * e) + lambda*(U_extended(i)^2);
    end
end
