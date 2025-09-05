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

% Outputs: we want to regulate the cart position and the pendulum angle.
C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

sys_ss = ss(A, B, C, D);

% Convert the continuous-time system to discrete-time for MPC
Ts = 0.1;  % Sampling time (seconds)
sys_d = c2d(sys_ss, Ts);

% Create an MPC controller object with the discrete model and sampling time
mpc_controller = mpc(sys_d, Ts);

% Set the prediction and control horizons for MPC
mpc_controller.PredictionHorizon = 20;
mpc_controller.ControlHorizon = 5;

% Set weighting parameters for MPC:
mpc_controller.Weights.ManipulatedVariables = 0.1;         % Weight on control input magnitude
mpc_controller.Weights.ManipulatedVariablesRate = 0.1;       % Weight on rate of change of control input
mpc_controller.Weights.OutputVariables = [1 1];              % Weights for both outputs (cart position & pendulum angle)
mpc_controller.Weights.ECR = 100000;                         % High penalty for constraint relaxation

% Set physical constraints on the control input (for example, force limits)
mpc_controller.ManipulatedVariables(1).Min = -10;
mpc_controller.ManipulatedVariables(1).Max = 10;

% Define simulation parameters and reference trajectory
T_sim = 10;                        % Total simulation time (seconds)
r = [zeros(10,2); ones(91,2)];       % Reference: 0 for first 1 second, then step to 1 for both outputs

% Simulate the MPC controller's closed-loop response
[yp, t, u,  F] = sim(mpc_controller, T_sim/Ts, r, [] );

% Plot the closed-loop responses for each output
figure;
subplot(2,1,1);
plot(t, yp(:,1), 'b', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Cart Position');
title('Closed-Loop Response (Cart Position)');
grid on;

subplot(2,1,2);
plot(t, yp(:,2), 'r', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Pendulum Angle');
title('Closed-Loop Response (Pendulum Angle)');
grid on;

% Plot the control input from the MPC
figure;
plot(t, F);
xlabel('Time (s)');
ylabel('Control Input (Wheels)');
title('Control Input (MPC Output)');
grid on;
