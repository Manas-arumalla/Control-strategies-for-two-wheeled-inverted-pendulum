clc; clear; close all;

%% 1. Plant Parameters (your values)
m_w = 0.432;      % mass of each wheel [kg]
m_p = 5.0;        % mass of the pendulum [kg]
I_w = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;      % Length of the pendulum [m]
r   = 0.0726;   % wheel radius [m]
g   = 9.81;     % acceleration due to gravity [m/s^2]

%% 2. Derived constants & original 4‑state model
a     = 2*m_w + m_p + 2*I_w/r^2;
b     = m_p * l;
c     = m_p * l^2 + I_p;
denom = a*c - b^2;

A_orig = [ 0, 1,               0,              0;
           0, 0,      - (b*m_p*g*l)/denom,      0;
           0, 0,               0,              1;
           0, 0,      (a*m_p*g*l)/denom,        0 ];
       
B_orig = [0; c/denom; 0; -b/denom];

%% 3. GA setup for augmentation + LQR tuning
% p = [alpha, delta, Q1, Q2, Q3, Q4, Q5, R]
lb = [0.1,  0.001,  10,   0.1,  10,   0.1,  1,    0.001];
ub = [5,    1,     1000, 100,  1000, 100,  100,  10   ];

opts = optimoptions('ga', ...
    'Display','iter', ...
    'PopulationSize',60, ...
    'MaxGenerations',250, ...
    'UseParallel',false);

costFun = @(p) costFunctionAugLQR(p, a, b, c, denom);

[p_opt, fval] = ga(costFun, 8, [],[],[],[], lb, ub, [], opts);

%% 4. Display optimized values
alpha_opt = p_opt(1);
delta_opt = p_opt(2);
Q_weights = p_opt(3:7);
R_opt     = p_opt(8);

fprintf('\nOptimized Parameters:\n');
fprintf(' alpha = %.3f, delta = %.4f\n', alpha_opt, delta_opt);
fprintf(' Q = diag([%.1f, %.1f, %.1f, %.1f, %.1f])\n', Q_weights);
fprintf(' R = %.4f\n', R_opt);

%% 5. Re‑simulate and plot with the best controller
A_aug = [ A_orig,               zeros(4,1);
          0,0, alpha_opt, 1,   -alpha_opt ];
B_aug = [B_orig; delta_opt];

Q_aug = diag(Q_weights);
K_opt = lqr(A_aug, B_aug, Q_aug, R_opt);
A_cl  = A_aug - B_aug*K_opt;

% simulate with ode45
z0    = [0;0;0.1;0;0.1];
tspan = [0 10];
[tt, zz] = ode45(@(t,z) A_cl*z, tspan, z0);

x      = zz(:,1);
theta  = zz(:,3);
z5     = zz(:,5);
u      = - (K_opt * zz')';

figure;
subplot(3,1,1);
plot(tt, x,    'LineWidth',2); ylabel('x (m)');
title('Cart Position');
grid on;
subplot(3,1,2);
plot(tt, theta,'r','LineWidth',2); ylabel('\theta (rad)');
title('Pendulum Angle');
grid on;
subplot(3,1,3);
plot(tt, z5,   'g','LineWidth',2); ylabel('z_5');
title('Auxiliary State');
grid on;

figure;
plot(tt, u, 'k','LineWidth',2);
xlabel('Time (s)'); ylabel('F (N)');
title('Control Input');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Cost Function ---
function J = costFunctionAugLQR(p, a, b, c, denom)
    % unpack
    alpha = p(1);  delta = p(2);
    Q1 = p(3); Q2 = p(4); Q3 = p(5); Q4 = p(6); Q5 = p(7);
    R  = p(8);
    
    % build augmented model
    A_aug = [ 0, 1,              0,              0,    0;
              0, 0,  - (b*5*9.81*0.5)/denom,     0,    0;
              0, 0,              0,              1,    0;
              0, 0,   (a*5*9.81*0.5)/denom,      0,    0;
              0, 0,         alpha,              1, -alpha ];
    B_aug = [0; c/denom; 0; -b/denom; delta];
    
    Q = diag([Q1,Q2,Q3,Q4,Q5]);
    
    % try LQR
    try
        K = lqr(A_aug, B_aug, Q, R);
    catch
        J = 1e8; return;
    end
    
    % closed-loop
    A_cl = A_aug - B_aug*K;
    if any(real(eig(A_cl)) >= 0)
        J = 1e8; return;
    end
    
    % simulate
    T  = 10; dt = 0.01;
    t  = 0:dt:T;
    z  = zeros(5,length(t));
    z(:,1) = [0;0;0.1;0;0.1];
    for k=1:length(t)-1
        z(:,k+1) = z(:,k) + dt*(A_cl*z(:,k));
    end
    
    % output errors
    y = [z(1,:); z(3,:)];      % x and theta
    e = y.^2;                  % squared error
    J = trapz(t, sum(e,1));    % integrated
    % penalty if not settled
    if norm(y(:,end))>0.05
        J = J + 1e5;
    end
end
