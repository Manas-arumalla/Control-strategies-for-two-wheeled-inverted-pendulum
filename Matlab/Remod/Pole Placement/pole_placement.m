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

C = [1, 0, 0, 0; 
     0, 0, 1, 0];  
D = [0; 0];

sys_ss = ss(A, B, C, D);

%% Pole Placement Design
% Choose desired closed-loop poles (modify these to tune performance)
desired_poles = [-5, -4, -3, -2];

% Compute the state feedback gain using pole placement:
Kp = place(A, B, desired_poles);
disp('Pole Placement Gain Kp:');
disp(Kp);

% Form the closed-loop system:
Ac_pp = A - B*Kp;
disp('Closed-loop eigenvalues with pole placement:');
eig_pp = eig(Ac_pp)

sys_cl_pp = ss(Ac_pp, B, C, D);

%% Simulation of Closed-Loop Response using Pole Placement
x0 = [0; 0; 0.1; 0];   % initial condition (perturbed state)
t = 0:0.01:5;           % simulation time vector

[y_pp, t, x_pp] = initial(sys_cl_pp, x0, t);

figure;
subplot(2,1,1);
plot(t, y_pp(:,1), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('Closed-Loop Response (Pole Placement)');
grid on;

subplot(2,1,2);
plot(t, y_pp(:,2), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

%% Frequency Domain Analysis (for example, from force F to pendulum angle)
sys_tf = tf(sys_cl_pp);
% Assume transfer function of interest is from the input F to pendulum angle (second output)
sys_tf2 = sys_tf(2); 

figure;
rlocus(sys_tf2);
title('Root Locus (Pole Placement)');
grid on;

figure;
nyquist(sys_tf2);
title('Nyquist Plot (Pole Placement)');
grid on;

figure;
bode(sys_tf2);
title('Bode Plot (Pole Placement)');
grid on;

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = y_pp(:,1);  % Cart/Wheel Position
angle_response = y_pp(:,2);     % Pendulum Angle

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
