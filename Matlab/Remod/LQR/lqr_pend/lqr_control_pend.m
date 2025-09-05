% --- 1) SYSTEM PARAMETERS ---
m_w = 1.0;          % mass of each wheel [kg]
m_p = 2.5;            % pendulum mass [kg]
r   = 0.05;         % wheel radius [m]
l   = 0.58;            % pendulum length [m]
g   = 9.81;           % gravity [m/s^2]
I_w = 0.5 * m_w*r^2;  % wheel inertia [kg·m²]
I_p = m_p*l^2;        % pendulum inertia [kg·m²]

% --- 2) BUILD A, B MATRICES (linearized around upright) ---
a     = 2*m_w + m_p + 2*I_w/r^2;
b     = m_p * l;
c_val = m_p * l^2 + I_p;
d     = m_p * g * l;
Delta = a*c_val - b^2;

A = [ 0,                            1,      0,           0;
      0,                            0, -b*d/Delta,       0;
      0,                            0,      0,           1;
      0,                            0,  a*d/Delta,       0 ];
B = [ 0;
      c_val/Delta;
      0;
     -b/Delta ];

% --- 3) DESIGN ANGLE‑ONLY LQR ---
C_angle = [0 0 1 1];           % pick only the pendulum‑angle state

% your A, B as before


% 1) Define Q so only state-3 is significant
q_phi = 9786.743;     % your previous angle weight
eps   = 1e-3;         % tiny weight to keep Q full rank
Q     = diag([eps, eps, q_phi, eps]);

% 2) Define R (must be > 0)
R     = 0.001;

% 3) Compute LQR gain
K = lqr(A, B, Q, R);

disp('Angle‑only LQR gain K:');
disp(K);

% 4) (Optional) simulate closed-loop pendulum angle
Ac     = A - B*K;
C_phi  = [0 0 1 0];             % output = pendulum angle
sys_cl = ss(Ac, B, C_phi, 0);
x0     = [0;0;0.1;0];           % initial tilt
t      = 0:0.01:5;
[phi,~,~] = initial(sys_cl, x0, t);

figure;
plot(t, phi, 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Pendulum Angle φ (rad)');
title('Pendulum Angle with Angle‑Only LQR');
grid on;

