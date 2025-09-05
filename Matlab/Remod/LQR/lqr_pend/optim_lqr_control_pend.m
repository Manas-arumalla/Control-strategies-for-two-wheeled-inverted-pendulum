%% angle_only_lqr_ga.m
clc; clear all; close all;

%% 1) PLANT PARAMETERS
m_w = 1.0;          % mass of each wheel [kg]
m_p = 2.5;            % pendulum mass [kg]
r   = 0.05;         % wheel radius [m]
l   = 0.58;            % pendulum length [m]
g   = 9.81;           % gravity [m/s^2]
I_w = 0.5 * m_w*r^2;  % wheel inertia [kg·m²]
I_p = m_p*l^2;        % pendulum inertia [kg·m²]

% Derived linearized model constants
a     = 2*m_w + m_p + 2*I_w/r^2;
b     = m_p * l;
c_val = m_p * l^2 + I_p;
d     = m_p * g * l;
Delta = a*c_val - b^2;

% State‑space A, B
A = [0, 1,          0,          0;
     0, 0,   -b*d/Delta,       0;
     0, 0,          0,          1;
     0, 0,    a*d/Delta,       0];
B = [0; c_val/Delta; 0; -b/Delta];

% We only output the pendulum angle
C = [0 0 1 0];
D = 0;

%% 2) GA SETUP: optimize q_phi only (weight on φ)
% Decision var p = q_phi
lb = 1e-2;     % lower bound on q_phi
ub = 1e5;      % upper bound on q_phi

% GA options
opts = optimoptions('ga', ...
    'Display','iter', ...
    'PopulationSize',30, ...
    'MaxGenerations',50, ...
    'UseParallel',false, ...
    'FunctionTolerance',1e-3);

% Run GA
[p_opt, fval] = ga(@(q) cost_qphi(q, A, B, C), 1, [], [], [], [], lb, ub, [], opts);

q_phi = p_opt;
fprintf('\nOptimal pendulum‑angle weight q_phi = %.3f\n', q_phi);
fprintf('Resulting settling time = %.3f s\n\n', fval);

%% 3) COMPUTE & SIMULATE ANGLE‑ONLY LQR WITH q_phi
eps   = 1e-3;                     % tiny weights on other states
Q     = diag([eps, eps, q_phi, eps]);
R     = 0.001;

K = lqr(A, B, Q, R);
disp('LQR gain K ='), disp(K);

% Closed‑loop
Ac    = A - B*K;
sys_cl = ss(Ac, B, C, D);

% Simulate free response from small initial tilt
x0    = [0;0;0.1;0];
t     = 0:0.01:5;
[phi, ~] = initial(sys_cl, x0, t);

% Plot
figure;
plot(t, phi, 'LineWidth',1.5)
xlabel('Time (s)')
ylabel('Pendulum Angle φ (rad)')
title('Angle‑Only LQR Closed‑Loop Response')
grid on


%% --- Cost function for GA: settling time of φ ---
function Tset = cost_qphi(q_phi, A, B, C)
    % tiny eps on unwatched states
    eps = 1e-3;
    Q   = diag([eps, eps, q_phi, eps]);
    R   = 0.001;
    
    % robustly handle bad Q
    try
        K = lqr(A, B, Q, R);
    catch
        Tset = 1e3; return
    end
    
    Ac = A - B*K;
    % check stability
    if any(real(eig(Ac)) >= 0)
        Tset = 1e3; return
    end
    
    sys = ss(Ac, B, C, 0);
    x0 = [0;0;0.1;0];
    tspan = 0:0.01:5;
    phi = initial(sys, x0, tspan);
    
    % settling time to within tol
    tol = 0.01;
    idx = find(abs(phi) > tol);
    if isempty(idx)
        Tset = 0;
    else
        Tset = tspan(max(idx));
    end
    % penalize if too slow
    if Tset >= tspan(end) - 0.1
        Tset = 1e3;
    end
end
