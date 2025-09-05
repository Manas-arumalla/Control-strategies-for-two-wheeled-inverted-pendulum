%% Define Plant Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % Length of the pendulum [m]
r   = 0.0726;     % wheel radius [m]
g   = 9.81;       % gravitational acceleration [m/s^2]

% Derived parameters from the linearized model:
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

Delta = a*c - b^2;  % determinant

%% Construct the Linearized State–Space Model
% States: [cart_position; cart_velocity; pendulum_angle; pendulum_angular_velocity]
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

%% Define Simulation Parameters for MPC
T_sim = 10;  % Total simulation time (seconds)
N_sim = round(T_sim/Ts);
% Reference trajectory: zero for the first second, then a step to 1 for both outputs.
r = [zeros(10,2); ones(N_sim-10,2)];

%% Genetic Algorithm Optimization for MPC Tuning
% Decision vector: p = [PH, CH, Wmv, Wmvr, Wy1, Wy2, W_ECR]
% Bounds:
%   PH: Prediction Horizon, integer in [5, 50]
%   CH: Control Horizon, integer in [1, 20] (penalty if CH > PH)
%   Wmv: Weight on manipulated variable, [0.001, 10]
%   Wmvr: Weight on manipulated variable rate, [0.001, 10]
%   Wy1: Weight on output (cart position), [0.1, 10]
%   Wy2: Weight on output (pendulum angle), [0.1, 10]
%   W_ECR: Weight on constraint relaxation, [1e3, 1e6]
lb = [5, 1, 0.001, 0.001, 0.1, 0.1, 1e3];
ub = [50, 20, 10,    10,    10,  10, 1e6];

% First two decision variables are integers.
IntCon = [1, 2];

% Set GA options (increase max generations as needed)
options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',500);

% Define the cost function handle. We pass the discrete model, sampling time, simulation time, and reference.
costFun = @(p) costFunctionMPC(p, sys_d, Ts, T_sim, r);

% Run the GA
[p_opt, fval] = ga(costFun, 7, [], [], [], [], lb, ub, [], IntCon, options);

fprintf('\nOptimized MPC Parameters (GA):\n');
fprintf('Prediction Horizon (PH) = %d\n', round(p_opt(1)));
fprintf('Control Horizon (CH)    = %d\n', round(p_opt(2)));
fprintf('Wmv = %.4f\n', p_opt(3));
fprintf('Wmvr = %.4f\n', p_opt(4));
fprintf('Wy1 = %.4f\n', p_opt(5));
fprintf('Wy2 = %.4f\n', p_opt(6));
fprintf('W_ECR = %.4f\n', p_opt(7));

%% Create MPC Controller with Optimized Parameters and Simulate
PH_opt = round(p_opt(1));
CH_opt = round(p_opt(2));
mpc_controller = mpc(sys_d, Ts);
mpc_controller.PredictionHorizon = PH_opt;
mpc_controller.ControlHorizon = CH_opt;
mpc_controller.Weights.ManipulatedVariables = p_opt(3);
mpc_controller.Weights.ManipulatedVariablesRate = p_opt(4);
mpc_controller.Weights.OutputVariables = [p_opt(5), p_opt(6)];
mpc_controller.Weights.ECR = p_opt(7);
% Set physical constraints on the manipulated variable
mpc_controller.ManipulatedVariables(1).Min = -10;
mpc_controller.ManipulatedVariables(1).Max = 10;

% Simulate the closed-loop response using the optimized MPC
[yp, t, u, ~] = sim(mpc_controller, N_sim, r, []);

%% Plot the Closed-Loop Responses
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

figure;
plot(t, u, 'k', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Control Input (Force)');
title('MPC Control Input');
grid on;

%% --- Function Definitions ---
function cost = costFunctionMPC(p, sys_d, Ts, T_sim, r)
    % p = [PH, CH, Wmv, Wmvr, Wy1, Wy2, W_ECR]
    % Round PH and CH to integers.
    PH = round(p(1));
    CH = round(p(2));
    
    % Apply a penalty if the Control Horizon exceeds the Prediction Horizon.
    if CH > PH
        cost = 1e6;
        return;
    end
    
    Wmv = p(3);
    Wmvr = p(4);
    Wy1 = p(5);
    Wy2 = p(6);
    W_ECR = p(7);
    
    try
        % Create an MPC controller with the candidate parameters.
        mpc_controller = mpc(sys_d, Ts);
        mpc_controller.PredictionHorizon = PH;
        mpc_controller.ControlHorizon = CH;
        mpc_controller.Weights.ManipulatedVariables = Wmv;
        mpc_controller.Weights.ManipulatedVariablesRate = Wmvr;
        mpc_controller.Weights.OutputVariables = [Wy1, Wy2];
        mpc_controller.Weights.ECR = W_ECR;
        mpc_controller.ManipulatedVariables(1).Min = -10;
        mpc_controller.ManipulatedVariables(1).Max = 10;
        
        % Number of simulation steps.
        N = round(T_sim/Ts);
        
        % Simulate the closed-loop response.
        [yp, ~, ~, ~] = sim(mpc_controller, N, r, []);
        
        % Compute the tracking error (sum of squared errors for both outputs).
        err = r - yp;
        cost = sum(sum(err.^2));
        
        % If the simulation returns NaN or very large error, penalize.
        if isnan(cost) || isinf(cost)
            cost = 1e6;
        end
    catch
        cost = 1e6;
    end
end
