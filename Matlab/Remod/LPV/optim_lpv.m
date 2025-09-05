%% Optimize LPV Gain Scheduling Design via GA
% Decision vector: p = [Q1, Q2, Q3, Q4, R, p_min, p_max]
%   where Q = diag([Q1, Q2, Q3, Q4]) and R is the scalar weight for LQR.
%   p_min and p_max define the range of the scheduling parameter (pendulum angle deviation).
%
% The nonlinear constraint ensures that p_min < p_max.
%
% The cost function simulates the closed-loop LPV controlled system over a fixed time
% horizon using Euler integration and computes the integrated squared error for cart position
% and pendulum angle. The GA minimizes this cost.

clear; clc; close all;

%% Model Parameters (given)
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * m_w * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

% Common terms
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;  % determinant

%% Nominal State-Space Model (at p=0)
A = [ 0,     1,                      0,              0;
      0,     0,               -b*d/Delta,              0;
      0,     0,                      0,              1;
      0,     0,                a*d/Delta,              0];

B = [ 0;
      c/Delta;
      0;
     -b/Delta];

C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

%% Fixed LQR Design Settings at each operating point will use:
% The candidate Q and R from the decision vector p will be used.
% (For each scheduling parameter value, we modify the effective gravity using:
%  d_eff = m_p*g*l*cos(p_val) and update A(4,3) accordingly.)

%% GA Optimization Setup
% Decision vector: p = [Q1, Q2, Q3, Q4, R, p_min, p_max]
% Suggested bounds:
%   Q1: [1, 10000]
%   Q2: [1, 1000]
%   Q3: [1, 10000]
%   Q4: [1, 1000]
%   R: [0.001, 1]
%   p_min: [-0.2, -0.001]   (in radians)
%   p_max: [0.001, 0.2]      (in radians)
lb = [1, 1, 1, 1, 0.001, -0.2, 0.001];
ub = [10000, 1000, 10000, 1000, 1, -0.001, 0.2];

% No integer constraints here (all continuous)
options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);

% Cost function: it simulates the LPV closed-loop system and returns the integrated squared error.
% We pass model parameters and design constants.
T_sim = 5;           % simulation time [s]
dt = 0.01;           % simulation time step
numSched = 5;        % number of scheduling points to compute LQR gains
costFun = @(p) lpvCostFunction(p, A, B, C, D, m_p, g, l, a, Delta, T_sim, dt, numSched);

% Nonlinear constraint: p(6) < p(7)  (i.e., p_min < p_max)
nonlcon = @(p) deal(p(6) - p(7), []);  % c = p_min - p_max <= 0

[p_opt, fval] = ga(costFun, 7, [], [], [], [], lb, ub, nonlcon, options);

fprintf('\nOptimized Parameters:\n');
fprintf('Q = diag([%.2f, %.2f, %.2f, %.2f])\n', p_opt(1), p_opt(2), p_opt(3), p_opt(4));
fprintf('R = %.4f\n', p_opt(5));
fprintf('p_min = %.4f rad, p_max = %.4f rad\n', p_opt(6), p_opt(7));

%% Simulation with Optimized Parameters
% Build the scheduling range from the optimized p_min and p_max.
p_range_opt = linspace(p_opt(6), p_opt(7), numSched);
% Compute the LQR gains over the range.
K_schedule = zeros(numSched, size(B,2), size(A,1));
Q_opt = diag(p_opt(1:4));
R_opt = p_opt(5);
for i = 1:numSched
    p_val = p_range_opt(i);
    % Effective gravity modification: use d_eff = m_p*g*l*cos(p_val)
    d_eff = m_p * g * l * cos(p_val);
    A_lpv = A;
    A_lpv(4,3) = a * d_eff / Delta;
    K_schedule(i,:,:) = lqr(A_lpv, B, Q_opt, R_opt);
end
% Create interpolation function for gain.
K_interp = @(p_val) interp1(p_range_opt, squeeze(K_schedule), p_val, 'linear', 'extrap');

% Simulate closed-loop LPV controlled system.
time = 0:dt:T_sim;
x = zeros(4, length(time));
x(:,1) = [0; 0; 0.1; 0];  % initial condition with small pendulum deviation
u_store = zeros(1, length(time)-1);
for k = 1:length(time)-1
    % Scheduling parameter is taken as current pendulum angle x(3)
    p_current = x(3,k);
    K_current = K_interp(p_current);
    u_store(k) = -K_current * x(:,k);
    % Update state using Euler integration (nominal dynamics)
    x_dot = A*x(:,k) + B*u_store(k);
    x(:,k+1) = x(:,k) + dt*x_dot;
end

figure;
subplot(2,1,1);
plot(time, x(1,:), 'b', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Cart Position (m)');
title('Optimized LPV Controller Closed-Loop Response: Cart Position');
grid on;

subplot(2,1,2);
plot(time, x(3,:), 'r', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
title('Optimized LPV Controller Closed-Loop Response: Pendulum Angle');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Subfunction: LPV Cost Function ---
function J = lpvCostFunction(p, A, B, C, D, m_p, g, l, a, Delta, T_sim, dt, numSched)
    % p = [Q1, Q2, Q3, Q4, R, p_min, p_max]
    Q_vals = p(1:4);
    R_val = p(5);
    p_min = p(6);
    p_max = p(7);
    
    % Define Q and R matrices.
    Q_mat = diag(Q_vals);
    R_mat = R_val;
    
    % Create scheduling range.
    p_range = linspace(p_min, p_max, numSched);
    
    % Preallocate LQR gains.
    K_schedule = zeros(numSched, size(B,2), size(A,1));
    
    % Loop over scheduling points and compute LQR gains.
    for i = 1:numSched
        p_val = p_range(i);
        % For small pendulum angle deviations, effective gravity modification:
        d_eff = m_p * g * l * cos(p_val);
        A_lpv = A;
        A_lpv(4,3) = a * d_eff / Delta;
        try
            K_schedule(i,:,:) = lqr(A_lpv, B, Q_mat, R_mat);
        catch
            % If lqr fails, assign a high cost.
            J = 1e6;
            return;
        end
    end
    
    % Create a gain interpolation function.
    K_interp = @(p_val) interp1(p_range, squeeze(K_schedule), p_val, 'linear', 'extrap');
    
    % Simulation: use Euler integration over T_sim.
    time = 0:dt:T_sim;
    N = length(time);
    x = zeros(4, N);
    x(:,1) = [0; 0; 0.1; 0];  % initial condition (small pendulum deviation)
    J = 0;
    % Cost: accumulate squared error of both outputs.
    for k = 1:N-1
        % Use current pendulum angle as scheduling parameter.
        p_current = x(3,k);
        K_current = K_interp(p_current);
        u = -K_current * x(:,k);
        % Update state using nominal continuous-time dynamics (for simulation simplicity)
        x_dot = A*x(:,k) + B*u;
        x(:,k+1) = x(:,k) + dt*x_dot;
        % Compute output and tracking error (desired reference is zero for both outputs).
        y = C*x(:,k);
        e = y;  % reference is zero.
        J = J + sum(e.^2)*dt;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
