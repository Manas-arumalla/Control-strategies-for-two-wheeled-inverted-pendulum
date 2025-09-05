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
%   a*x_ddot + b*theta_ddot = F
%   b*x_ddot + c*theta_ddot = d*theta
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

% Create state-space plant (to be used in augmentation)
sys = ss(A, B, C, D);

%% H∞ Weighting Function Parameter Optimization Using GA
% We will optimize the parameters for the weighting functions used in the
% H∞ synthesis. The decision vector is:
%    p = [Wp_num, Wp_den, Wu]
%
%   where we form the performance weight as:
%     Wp(s) = (s + Wp_num)/(s + Wp_den)
%   and Wu is the control weight.
%
% Set lower and upper bounds:
lb = [0.1, 0.1, 0.001];
ub = [100, 50, 10];

% Set GA options (adjust PopulationSize and MaxGenerations as needed)
options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',300);

% Cost function handle: it returns a cost combining the achieved H∞ gamma and settling time.
costFun = @(p) costFunctionHinf(p, sys);

% Run GA optimization.
[p_opt, fval] = ga(costFun, 3, [], [], [], [], lb, ub, [], options);

fprintf('\nOptimized Weighting Parameters:\n');
fprintf('Wp_num = %.3f\n', p_opt(1));
fprintf('Wp_den = %.3f\n', p_opt(2));
fprintf('Wu     = %.3f\n', p_opt(3));

%% Synthesize H∞ Controller with Optimized Weights
s = tf('s');
Wp_opt = (s + p_opt(1))/(s + p_opt(2));
Wu_opt = p_opt(3);
% No weighting on uncertainty is used here, so we leave the second weight empty.
P_aug = augw(sys, Wp_opt, [], Wu_opt);

% Specify number of measured outputs and control inputs.
nmeas = 2;  % measured outputs: [cart position; pendulum angle]
ncon  = 1;  % one control input

% Synthesize the H∞ controller.
[K_hinf, CL, gamma, info] = hinfsyn(P_aug, nmeas, ncon);

fprintf('\nAchieved H∞ performance gamma: %.3f\n', gamma);

% Form the closed-loop system via LFT.
sys_cl_hinf = lft(P_aug, K_hinf);

%% Simulate Closed-Loop Step Response
t = 0:0.01:5;
[y_hinf, t_hinf, x_hinf] = step(sys_cl_hinf, t);

figure;
subplot(2,1,1);
plot(t_hinf, y_hinf(:,1), 'b', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Cart Position (m)');
title('H∞ Closed-Loop Step Response (Cart Position)');
grid on;
subplot(2,1,2);
plot(t_hinf, y_hinf(:,2), 'r', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Pendulum Angle (rad)');
title('H∞ Closed-Loop Step Response (Pendulum Angle)');
grid on;

%% Frequency Domain Analysis for the H∞ Closed-Loop
sys_tf = tf(sys_cl_hinf);
% Extract the transfer function from the control input to the pendulum angle (output 2)
G = sys_tf(2,1);

figure;
rlocus(G);
title('Root Locus (Force to Pendulum Angle)');
grid on;

figure;
nyquist(G);
title('Nyquist Plot (Force to Pendulum Angle)');
grid on;

figure;
bode(G);
title('Bode Plot (Force to Pendulum Angle)');
grid on;

%% --- Function Definitions ---
function cost = costFunctionHinf(p, sys)
    % p = [Wp_num, Wp_den, Wu]
    s = tf('s');
    Wp = (s + p(1))/(s + p(2));
    Wu = p(3);
    
    try
        % Build the augmented (generalized) plant.
        P = augw(sys, Wp, [], Wu);
        
        % Synthesize the H∞ controller.
        nmeas = 2;  % measured outputs: [cart position; pendulum angle]
        ncon = 1;   % one control input
        [~, ~, gamma, ~] = hinfsyn(P, nmeas, ncon);
        
        % Form the closed-loop system via LFT.
        sys_cl = lft(P, ss(zeros(ncon,nmeas)));  % here we consider the nominal closed-loop without K explicitly
        
        % Alternatively, simulate the closed-loop step response with the synthesized controller:
        % We re-synthesize the controller and then form the closed-loop system.
        [K, ~, ~, ~] = hinfsyn(P, nmeas, ncon);
        sys_cl = lft(P, K);
        
        % Simulate step response.
        t_sim = linspace(0, 5, 500);
        [y, ~] = step(sys_cl, t_sim);
        
        % Evaluate the settling time for the pendulum angle (assume second output)
        tol = 0.02;
        T_set = settlingTime(t_sim, y(:,2), tol);
        
        % Combine the achieved gamma and settling time to form the cost.
        % (You can adjust the relative weighting here.)
        cost = gamma + T_set;
        
        % Impose a heavy penalty if gamma is high or the system does not settle.
        if isnan(cost) || isinf(cost) || gamma > 1000 || T_set > 5
            cost = 1e6;
        end
    catch
        cost = 1e6;
    end
end

function T_set = settlingTime(t, signal, tol)
    % Compute the settling time as the last time when the absolute error from the final value exceeds tol.
    final_value = signal(end);
    err = abs(signal - final_value);
    idx = find(err > tol);
    if isempty(idx)
        T_set = t(1);
    else
        T_set = t(idx(end));
    end
    % If the settling time is near the end of simulation, impose a penalty.
    if T_set >= t(end) - 0.1
        T_set = 1e3;
    end
end
