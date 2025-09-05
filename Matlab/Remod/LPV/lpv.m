%% Model Parameters
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * m_w * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

% Common terms
a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

Delta = a*c - b^2; 

%% Define the nominal state-space model (at p = 0)
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

%% LQR design weighting matrices (used at each operating point)
Q = diag([100, 1, 1000, 1]);  
R = 0.01; 

%% LPV Gain Scheduling Design
% Define a range for the scheduling parameter p (here, pendulum angle deviation)
p_range = linspace(-0.1, 0.1, 5);  % for example, from -0.1 rad to 0.1 rad
numPoints = length(p_range);

% Preallocate array to store LQR gains for each operating point
K_schedule = zeros(numPoints, size(B,2), size(A,1));

% Loop over the scheduling parameter range and compute LQR gains
for i = 1:numPoints
    p = p_range(i);
    
    % Modify the gravitational effect based on the pendulum angle.
    % For small deviations, you might use: d_eff = m_p*g*l*cos(p)
    d_eff = m_p * g * l * cos(p);
    
    % Modify the A matrix accordingly. Here we assume that only the (4,3) element changes.
    A_lpv = A;
    A_lpv(4,3) = a * d_eff / Delta;
    
    % Compute LQR gain at this operating point
    K_schedule(i,:,:) = lqr(A_lpv, B, Q, R);
end

%% Create a Gain Interpolation Function
% This function returns the controller gain for any current value of the scheduling parameter p.
K_interp = @(p_val) interp1(p_range, squeeze(K_schedule), p_val, 'linear', 'extrap');

%% Simulation of Closed-Loop LPV Controlled System
% Simulation parameters
dt = 0.01;
T = 5;
time = 0:dt:T;

% Initial state: note that the third element is the pendulum angle deviation
x0 = [0; 0; 0.1; 0];  
x = zeros(4, length(time));
x(:,1) = x0;
u = zeros(1, length(time));

% For simulation, we use the nominal A matrix (you can also update A in the loop if desired)
% Here, we assume that the plant dynamics are affected by p just as in our gain design.
for k = 1:length(time)-1
    % The current scheduling parameter is taken as the pendulum angle deviation (x(3))
    p_current = x(3,k);
    
    % Interpolate to get the current controller gain K
    K_current = K_interp(p_current);
    
    % Control law: u = -K(x)*x
    u(k) = -K_current * x(:,k);
    
    % Update state (using Euler integration for simplicity)
    % For a more accurate simulation, consider using ODE solvers.
    x_dot = A * x(:,k) + B * u(k);
    x(:,k+1) = x(:,k) + dt * x_dot;
end

%% Plot Results
figure;
subplot(2,1,1);
plot(time, x(1,:), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('LPV Controller Closed-Loop Response');
grid on;

subplot(2,1,2);
plot(time, x(3,:), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = x(1,:);  % Cart/Wheel Position
angle_response = x(3,:);     % Pendulum Angle

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

