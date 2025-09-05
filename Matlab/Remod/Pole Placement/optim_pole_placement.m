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
% States: x = [cart_position; cart_velocity; pendulum_angle; pendulum_angular_velocity]
A = [ 0,     1,                      0,              0;
      0,     0,               -b*d/Delta,              0;
      0,     0,                      0,              1;
      0,     0,                a*d/Delta,              0];

B = [ 0;
      c/Delta;
      0;
     -b/Delta];

% Outputs: we want to regulate cart position and pendulum angle.
C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

%% Set Up GA Optimization for Pole Placement
% Decision vector: p = [p1, p2, p3, p4] (all positive) so that desired poles are -p.
% For example, choose bounds for each p_i in [0.1, 50]
lb = [0.1, 0.1, 0.1, 0.1];
ub = [50, 50, 50, 50];

options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',300);

% The cost function simulates the closed-loop response and returns the settling time (for the pendulum angle)
costFun = @(p) costFunctionPolePlacement(p, A, B, C, D);

[p_opt, fval] = ga(costFun, 4, [], [], [], [], lb, ub, [], options);

% Compute the desired poles from the optimized decision vector:
desired_poles_opt = -p_opt;
fprintf('\nOptimized Desired Poles:\n');
disp(desired_poles_opt);

%% Compute the Pole Placement Gain Using the Optimized Poles
Kp_opt = place(A, B, desired_poles_opt);
Ac_opt = A - B*Kp_opt;
sys_cl_opt = ss(Ac_opt, B, C, D);

fprintf('\nClosed-loop eigenvalues (should equal the desired poles):\n');
disp(eig(Ac_opt));

%% Simulate the Closed-Loop Response with the Optimized Pole Placement
x0 = [0; 0; 0.1; 0];  % initial condition (small pendulum deviation)
t = 0:0.01:5;        % simulation time vector

[y, t, x] = initial(sys_cl_opt, x0, t);

figure;
subplot(2,1,1);
plot(t, y(:,1), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('Closed-Loop Response with Optimized Pole Placement');
grid on;

subplot(2,1,2);
plot(t, y(:,2), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

%% --- Function Definitions ---

function cost = costFunctionPolePlacement(p, A, B, C, D)
    % p is a 4-element vector (all positive) so that the desired poles are -p.
    desired_poles = -p;
    
    try
        % Compute state-feedback gain using pole placement
        Kp = place(A, B, desired_poles);
        A_cl = A - B*Kp;
        sys_cl = ss(A_cl, B, C, D);
        
        % Simulate the closed-loop response from a small initial deviation.
        tspan = 0:0.01:5;
        x0 = [0; 0; 0.1; 0];
        y = initial(sys_cl, x0, tspan);
        
        % For cost, compute the settling time for the pendulum angle (2nd output)
        tol = 0.02;  % tolerance for settling
        T_set = settlingTime(tspan, y(:,2), tol);
        
        cost = T_set;  % we minimize settling time
        
        % Impose a heavy penalty if settling time is near the simulation horizon
        if T_set >= tspan(end)-0.1
            cost = 1e6;
        end
        
        % In case of NaN or error in simulation
        if isnan(cost) || isinf(cost)
            cost = 1e6;
        end
    catch
        cost = 1e6;
    end
end

function T_set = settlingTime(t, signal, tol)
    % Compute the settling time as the last time when the error (from the final value) exceeds tol.
    final_value = signal(end);
    err = abs(signal - final_value);
    idx = find(err > tol);
    if isempty(idx)
        T_set = t(1);
    else
        T_set = t(idx(end));
    end
    % If the settling time is very close to the end of simulation, impose a penalty.
    if T_set >= t(end)-0.1
        T_set = 1e3;
    end
end
