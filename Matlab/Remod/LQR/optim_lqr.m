clc;
clear all;
close all;

%% Define Plant Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 0.5 * (0.432) * (0.0726)^2;  % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                 % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;        % Length of the pendulum [m]
r   = 0.0726;     % wheel radius [m]
g   = 9.81;       % acceleration due to gravity [m/s^2]

% Derived parameters from the linearized model:
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2;  % determinant

% Construct the linearized state–space model
% States: [cart_position; cart_velocity; pendulum_angle; pendulum_angular_velocity]
A = [ 0,     1,                      0,              0;
      0,     0,               -b*d/Delta,              0;
      0,     0,                      0,              1;
      0,     0,                a*d/Delta,              0];

B = [ 0;
      c/Delta;
      0;
     -b/Delta];

% Outputs: regulate cart position and pendulum angle
C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

%% GA Optimization Setup for LQR Design
% Decision vector: p = [q1, q2, q3, q4, r_val]
% Q = diag(q1, q2, q3, q4) and R = r_val.
% Set lower bounds and upper bounds.
lb = [0, 0, 0, 0, 0.001];
ub = [10000, 10000, 10000, 10000, 10];

% Set GA options
options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);

% Define the cost function handle (see function below)
costFun = @(p) costFunctionLQR(p, A, B, C, D);

% Run GA optimization
[p_opt, fval] = ga(costFun, 5, [], [], [], [], lb, ub, [], options);

% Extract optimized parameters
q1 = p_opt(1);
q2 = p_opt(2);
q3 = p_opt(3);
q4 = p_opt(4);
r_val = p_opt(5);

fprintf('\nOptimized LQR Parameters:\n');
fprintf('q1 = %.3f, q2 = %.3f, q3 = %.3f, q4 = %.3f, R = %.3f\n', q1, q2, q3, q4, r_val);

%% Compute LQR Gain Using Optimized Q and R
Q_opt = diag([q1, q2, q3, q4]);
R_opt = r_val;
K = lqr(A, B, Q_opt, R_opt);
disp('Optimized LQR Gain Matrix K:');
disp(K);

% Form closed-loop system
Ac = A - B*K;
sys_cl = ss(Ac, B, C, D);

fprintf('\nClosed-loop Eigenvalues:\n');
disp(eig(Ac));

%% Simulate the Closed-Loop Response
x0 = [0; 0; 0.1; 0];  % initial condition (small pendulum deviation)
t = 0:0.01:5;    
[y, t, x] = initial(sys_cl, x0, t);

figure;
subplot(2,1,1);
plot(t, y(:,1), 'b', 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('Closed-Loop Response (Optimized LQR)');
grid on;

subplot(2,1,2);
plot(t, y(:,2), 'r', 'LineWidth', 1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

%% Frequency Domain Analysis
sys_tf = tf(sys_cl);
% Assume transfer function of interest is from the input (F) to pendulum angle (second output)
sys_tf1 = sys_tf(2); 

figure;
rlocus(sys_tf1);
title('Root Locus');
grid on;

figure;
nyquist(sys_tf1);
title('Nyquist Plot');
grid on;

figure;
bode(sys_tf1);
title('Bode Plot');
grid on;

% Step response metrics for the pendulum angle channel
info_pendulum = stepinfo(sys_tf);
fprintf('\nPendulum Angle Step Response Metrics:\n');
fprintf('Rise Time: %.3f seconds\n', info_pendulum.RiseTime);
fprintf('Settling Time: %.3f seconds\n', info_pendulum.SettlingTime);

%% --- Function Definitions ---

function cost = costFunctionLQR(p, A, B, C, D)
    % p = [q1, q2, q3, q4, r_val]
    Q = diag(p(1:4));
    R = p(5);
    try
        % Compute LQR gain
        K = lqr(A, B, Q, R);
        Ac = A - B*K;
        % Penalize if the closed-loop system is unstable.
        if any(real(eig(Ac)) >= 0)
            cost = 1e6;
            return;
        end
        sys_cl = ss(Ac, B, C, D);
        % Simulate the closed-loop response
        tspan = linspace(0, 5, 500);
        x0 = [0; 0; 0.1; 0];
        y = initial(sys_cl, x0, tspan);
        % Calculate settling times for cart position and pendulum angle.
        tol = 0.02;
        T_set_cart = settlingTime(tspan, y(:,1), tol);
        T_set_pend = settlingTime(tspan, y(:,2), tol);
        cost = T_set_cart + T_set_pend;
        % If simulation does not settle, impose a high penalty.
        if isnan(cost) || isinf(cost)
            cost = 1e6;
        end
    catch
        cost = 1e6;
    end
end

function T_set = settlingTime(t, signal, tol)
    % Compute settling time as the last time the absolute error from the final value exceeds tol.
    final_value = signal(end);
    err = abs(signal - final_value);
    idx = find(err > tol);
    if isempty(idx)
        T_set = t(1);
    else
        T_set = t(idx(end));
    end
    % Penalize if the settling time is too close to the end of simulation.
    if T_set >= t(end) - 0.1
        T_set = 1e3;
    end
end
