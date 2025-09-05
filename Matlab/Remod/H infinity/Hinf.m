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

sys = ss(A, B, C, D);

%% Define Weighting Functions for H∞ Design
s = tf('s');
Wp = (s+1.1)/(s+46.450);  % Performance weight (shaping tracking error)
Wu = 0.299;             % Control weight

% Form the augmented (generalized) plant.
% (Here we use MATLAB's augw function; adjust if further augmentation is needed.)
P = augw(sys, Wp, [], Wu);

%% H∞ Synthesis
% Specify number of measured outputs and control inputs.
nmeas = 2;  % measured outputs: [cart position; pendulum angle]
ncon  = 1;  % one control input (force F)

% Synthesize the H∞ controller.
[K_hinf, CL, gamma, info] = hinfsyn(P, nmeas, ncon);

% Save the controller matrices for later use if desired.
[A_hinf, B_hinf, C_hinf, D_hinf] = ssdata(K_hinf);
save('hinf_controller.mat', 'A_hinf', 'B_hinf', 'C_hinf', 'D_hinf');

% Display the H∞ controller matrices.
disp('H-infinity controller (K_hinf):');
disp(K_hinf);
disp('Achieved H-infinity performance gamma:');
disp(gamma);
disp('Controller matrix A_hinf:');
disp(A_hinf);
disp('Controller matrix B_hinf:');
disp(B_hinf);
disp('Controller matrix C_hinf:');
disp(C_hinf);
disp('Controller matrix D_hinf:');
disp(D_hinf);

%% Form the Closed-Loop System via LFT
sys_cl_hinf = lft(P, K_hinf);
disp('Closed-loop eigenvalues (H-infinity):');
disp(eig(sys_cl_hinf));

%% Simulation: Step Response
% Simulate a step response for the closed-loop system.
t = 0:0.01:5;
[y_hinf, t_hinf, x_hinf] = step(sys_cl_hinf, t);

figure;
subplot(2,1,1);
plot(t_hinf, y_hinf(:,2), 'b', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Cart Position (m)');
title('H∞ Closed-Loop Step Response (Cart Position)');
grid on;
subplot(2,1,2);
plot(t_hinf, y_hinf(:,1), 'r', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Pendulum Angle (rad)');
title('H∞ Closed-Loop Step Response (Pendulum Angle)');
grid on;

%% Frequency Domain Analysis
% Convert the closed-loop system to a transfer function model.
sys_tf = tf(sys_cl_hinf);
% For a SIMO system, extract the transfer function from the single input (column 1)
% to the second output (pendulum angle) using two subscripts.
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

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = y_hinf(:,1);  % Cart/Wheel Position
angle_response = y_hinf(:,2);     % Pendulum Angle

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
