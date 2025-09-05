%% Optimize ASLO Controller Parameters via GA
% Decision vector: p = [alpha, delta, Q1, Q2, Q3, Q4, Q5, R]
% where Q_aug = diag([Q1, Q2, Q3, Q4, Q5]) and R is the scalar.
% The cost function simulates the closed-loop system and returns the integrated
% squared error (focusing on cart position and pendulum angle).
%
% Adjust bounds and cost weights as needed.

clear; clc; close all;

%% 1. Model Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

%% 2. Derived Constants
a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;
denom = a*c - b^2;

%% 3. Augmentation Tuning (to be optimized)
% The auxiliary dynamics are:
%   z5_dot = theta_dot + alpha*(theta - z5) + delta*F
% We'll optimize over alpha and delta.
% Default values were: alpha = 1, delta = 0.05.

%% 4. Nominal Model (4-state)
A_orig = [0, 1, 0, 0;
          0, 0, - (b*m_p*g*l)/denom, 0;
          0, 0, 0, 1;
          0, 0, (a*m_p*g*l)/denom, 0];
B_orig = [0; c/denom; 0; -b/denom];

%% 5. Augmented Model
% Augmented state: z = [x; x_dot; theta; theta_dot; z5]
% with auxiliary dynamics:
%    z5_dot = theta_dot + alpha*(theta - z5) + delta*F
% The augmented A and B matrices depend on alpha and delta.
% We'll compute them inside the cost function.
%
% The structure is:
% A_aug = [ A_orig,    zeros(4,1);
%           [0 0 alpha 1],   -alpha ];
% B_aug = [ B_orig;
%           delta ];
%
% For clarity, here we show the template:
%
% A_aug = [ 0,   1,              0,              0,      0;
%           0,   0,  - (b*m_p*g*l)/denom,       0,      0;
%           0,   0,              0,              1,      0;
%           0,   0,   (a*m_p*g*l)/denom,         0,      0;
%           0,   0,          alpha,              1,   -alpha ];
%
% B_aug = [0; c/denom; 0; -b/denom; delta];

%% 6. GA Optimization Setup
% Decision vector: p = [alpha, delta, Q1, Q2, Q3, Q4, Q5, R]
% Suggested bounds:
%   alpha: [0.1, 5]
%   delta: [0.001, 1]
%   Q1: [10, 1000]      (weight on x)
%   Q2: [0.1, 100]      (weight on x_dot)
%   Q3: [10, 1000]      (weight on theta)
%   Q4: [0.1, 100]      (weight on theta_dot)
%   Q5: [1, 100]        (weight on z5)
%   R: [0.001, 10]
lb = [0.1, 0.001, 10, 0.1, 10, 0.1, 1, 0.001];
ub = [5, 1, 1000, 100, 1000, 100, 100, 10];

options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
costFun = @(p) costFunctionASLO(p, m_p, g, l, a, b, c, denom);

[p_opt, fval] = ga(costFun, 8, [], [], [], [], lb, ub, [], options);

fprintf('\nOptimized Parameters:\n');
fprintf('alpha = %.3f, delta = %.4f\n', p_opt(1), p_opt(2));
fprintf('Q = diag([%.2f, %.2f, %.2f, %.2f, %.2f])\n', p_opt(3), p_opt(4), p_opt(5), p_opt(6), p_opt(7));
fprintf('R = %.4f\n', p_opt(8));

%% 7. Simulation with Optimized Parameters
alpha_opt = p_opt(1);
delta_opt = p_opt(2);
Q_aug = diag(p_opt(3:7));
R_opt = p_opt(8);

% Build augmented matrices with optimized alpha and delta.
A_aug = [ 0,   1,                     0,                    0,      0;
          0,   0,         - (b*m_p*g*l)/denom,              0,      0;
          0,   0,                     0,                    1,      0;
          0,   0,         (a*m_p*g*l)/denom,                0,      0;
          0,   0,                 alpha_opt,                 1,   -alpha_opt];
      
B_aug = [0; c/denom; 0; -b/denom; delta_opt];

% Compute LQR gain for the augmented system.
K_opt = lqr(A_aug, B_aug, Q_aug, R_opt);

% Closed-loop dynamics:
A_cl = A_aug - B_aug*K_opt;

% Simulation setup
T_sim = 10;    % simulation time (s)
tspan = [0 T_sim];
% Initial condition: small deviation in theta and z5; x and x_dot start at 0.
z0 = [0; 0; 0.1; 0; 0.1];

[t, z_sol] = ode45(@(t,z) A_cl*z, tspan, z0);

% Extract states
x_sim = z_sol(:, 1:4);
z5_sim = z_sol(:, 5);
% Reconstruct control input: u = -K_opt * z
u_sim = - (z_sol * K_opt');

%% Plot Results
figure;
subplot(3,1,1);
plot(t, x_sim(:,1), 'b', 'LineWidth',2);
xlabel('Time (s)'); ylabel('Cart Position (m)');
title('Optimized ASLO: Cart Position');
grid on;

subplot(3,1,2);
plot(t, x_sim(:,3), 'r', 'LineWidth',2);
xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
title('Optimized ASLO: Pendulum Angle');
grid on;

subplot(3,1,3);
plot(t, u_sim, 'k', 'LineWidth',2);
xlabel('Time (s)'); ylabel('Control Input (N)');
title('Optimized ASLO: Control Input');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Subfunction: Cost Function for ASLO Optimization ---
function J = costFunctionASLO(p, mp, g, l, a, b, c, denom)
    % p = [alpha, delta, Q1, Q2, Q3, Q4, Q5, R]
    alpha = p(1);
    delta = p(2);
    Q1 = p(3); Q2 = p(4); Q3 = p(5); Q4 = p(6); Q5 = p(7);
    R_val = p(8);
    
    % Build the augmented system matrices:
    % A_aug and B_aug depend on alpha and delta.
    A_aug = [ 0,   1,                      0,                      0,      0;
              0,   0,         - (b*mp*g*l)/denom,                0,      0;
              0,   0,                      0,                      1,      0;
              0,   0,         (a*mp*g*l)/denom,                  0,      0;
              0,   0,                  alpha,                     1,   -alpha];
    B_aug = [0; c/denom; 0; -b/denom; delta];
    
    % Define Q and R matrices for LQR design.
    Q_mat = diag([Q1, Q2, Q3, Q4, Q5]);
    R_mat = R_val;
    
    % Compute LQR gain.
    try
        K = lqr(A_aug, B_aug, Q_mat, R_mat);
    catch
        J = 1e6;
        return;
    end
    
    % Closed-loop dynamics matrix.
    A_cl = A_aug - B_aug*K;
    
    % Simulation settings:
    T_sim = 10;      % simulation time in seconds
    dt = 0.01;       % time step for simulation
    t = 0:dt:T_sim;
    
    % Initial condition for the augmented state.
    % [x; x_dot; theta; theta_dot; z5]
    z0 = [0; 0; 0.1; 0; 0.1];
    
    % Simulate the closed-loop system using Euler integration.
    N = length(t);
    z = zeros(5, N);
    z(:,1) = z0;
    for k = 1:N-1
        z(:,k+1) = z(:,k) + dt * A_cl * z(:,k);
    end
    
    % Define the outputs: we focus on cart position (state 1) and pendulum angle (state 3).
    y = [z(1,:); z(3,:)];
    % Desired reference is zero.
    ref = zeros(size(y));
    
    % Compute cost as the integrated squared error.
    J = trapz(t, sum((ref - y).^2, 1));
    
    % Penalize if the error remains high at the end (indicating poor settling).
    if norm(y(:,end)) > 0.05
        J = J + 1e4;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
