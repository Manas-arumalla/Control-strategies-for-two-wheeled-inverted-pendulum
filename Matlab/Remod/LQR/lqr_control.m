m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;

Delta = a*c - b^2; 

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

Q = diag([9786.743, 0.754, 0.626, 1.224]);  
R = 0.001;

K = lqr(A, B, Q, R);

disp('LQR gain matrix K:');
disp(K);

Ac = A - B*K;
sys_cl = ss(Ac, B, C, D);

disp('Closed-loop eigenvalues:');
disp(eig(Ac));

x0 = [0; 0; 0.1; 0];
t = 0:0.01:5;    

[y, t, x] = initial(sys_cl, x0, t);

figure;
subplot(2,1,1);
plot(t, y(:,1), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('Closed-Loop Response (LQR)');
grid on;

subplot(2,1,2);
plot(t, y(:,2), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;

sys_tf = tf(sys_cl);
% sys_tf(2) corresponds to the pendulum angle output
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

% Desired reference is zero for both cart position and pendulum angle
desired_reference = 0;
tolerance = 0.01;  % Tolerance for settling time

% Responses
position_response = y(:,1);  % Cart/Wheel Position
angle_response = y(:,2);     % Pendulum Angle

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

