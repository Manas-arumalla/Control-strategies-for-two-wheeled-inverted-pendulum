%% iLQR for a Segway-like Nonlinear System (Reduced Plots)
% This script implements an iterative LQR (iLQR) controller for the 
% nonlinear dynamics of a two-wheel inverted pendulum system (Segway)
% starting with a pendulum angle of 0.1 rad. The goal is to stabilize the
% wheel position (x) and the pendulum angle (θ) to zero.
%
% Only the wheel position and pendulum angle are plotted in the final results.

clear; clc; close all;

%% System Parameters
% Wheel and pendulum parameters
m_w = 0.432;            % mass of each wheel [kg]
m_p = 5.0;              % mass of the pendulum [kg]
I_w = 1/2 * m_w * (0.0726)^2;    % moment of inertia of each wheel [kg*m^2]
I_p = 5 * (0.4)^2;               % moment of inertia of the pendulum [kg*m^2]
l   = 0.4;              % distance from axle to pendulum center of mass [m]
r   = 0.0726;           % wheel radius [m]
g   = 9.81;             % gravitational acceleration [m/s^2]

% Derived parameters
a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;

%% Simulation and iLQR Setup
dt = 0.01;              % time step [s]
T  = 5;                 % total simulation time [s]
N  = round(T/dt);       % number of time steps

% Cost function weights (tune these as needed)
% Here we penalize deviation in x (wheel position) and theta (pendulum angle)
Q   = diag([100, 1, 1000, 10]);   % running state cost
R   = 0.001;                      % running control cost
Qf  = diag([1000, 10, 10000, 100]); % terminal state cost

% Desired state: upright equilibrium (x = 0, theta = 0)
x_target = [0; 0; 0; 0];

% Initial state: start with a 0.1 rad deviation in pendulum angle,
% zero velocities and wheel position
x0 = [0; 0; 0.1; 0];

% Initial control sequence (zeros)
u_seq = zeros(1, N);

max_iter = 200;        % maximum iLQR iterations
tol = 1e-6;           % convergence tolerance on cost change

%% iLQR Main Loop
for iter = 1:max_iter
    %--- Forward rollout with current control sequence ---
    x_traj = zeros(4, N+1);
    x_traj(:,1) = x0;
    for k = 1:N
        % RK4 integration for improved accuracy
        x_traj(:,k+1) = rk4(@(x,u) dynamics(x,u,a,b,c,m_p,g,l), x_traj(:,k), u_seq(k), dt);
    end
    
    %--- Compute total cost for the current trajectory ---
    cost = 0;
    for k = 1:N
        dx = x_traj(:,k) - x_target;
        cost = cost + dx'*Q*dx + u_seq(k)'*R*u_seq(k);
    end
    dx = x_traj(:,end) - x_target;
    cost = cost + dx'*Qf*dx;
    
    fprintf('Iteration %d: Cost = %.3f\n', iter, cost);
    
    %--- Linearize dynamics along trajectory using finite differences ---
    A_seq = zeros(4,4,N);
    B_seq = zeros(4,1,N);
    eps_fd = 1e-5;
    for k = 1:N
        xk = x_traj(:,k);
        uk = u_seq(k);
        A = zeros(4,4);
        for i = 1:4
            dx_pert = zeros(4,1);
            dx_pert(i) = eps_fd;
            f_plus  = dynamics(xk + dx_pert, uk, a,b,c,m_p,g,l);
            f_minus = dynamics(xk - dx_pert, uk, a,b,c,m_p,g,l);
            A(:,i) = (f_plus - f_minus)/(2*eps_fd);
        end
        f_plus  = dynamics(xk, uk + eps_fd, a,b,c,m_p,g,l);
        f_minus = dynamics(xk, uk - eps_fd, a,b,c,m_p,g,l);
        B = (f_plus - f_minus)/(2*eps_fd);
        
        % Discrete time approximation: x_{k+1} = x_k + dt*f(x,u)
        A_seq(:,:,k) = eye(4) + A*dt;
        B_seq(:,:,k) = B*dt;
    end
    
    %--- Backward Pass ---
    Vx = Qf*(x_traj(:,end) - x_target);
    Vxx = Qf;
    k_seq = zeros(1, N);        % feedforward control corrections
    K_seq = zeros(1,4, N);       % feedback gain matrices
    reg = 1e-6;                 % regularization for numerical stability
    
    for k = N:-1:1
        Qx  = Q*(x_traj(:,k) - x_target) + A_seq(:,:,k)'*Vx;
        Qu  = R*u_seq(k) + B_seq(:,:,k)'*Vx;
        Qxx = Q + A_seq(:,:,k)'*Vxx*A_seq(:,:,k);
        Quu = R + B_seq(:,:,k)'*Vxx*B_seq(:,:,k);
        Qux = B_seq(:,:,k)'*Vxx*A_seq(:,:,k);
        
        % Regularization
        Quu = Quu + reg*eye(size(Quu));
        
        invQuu = inv(Quu);
        k_seq(k) = -invQuu * Qu;
        K_seq(:,:,k) = -invQuu * Qux;
        
        Vx  = Qx + K_seq(:,:,k)'*Quu*k_seq(k) + K_seq(:,:,k)'*Qu + Qux'*k_seq(k);
        Vxx = Qxx + K_seq(:,:,k)'*Quu*K_seq(:,:,k) + K_seq(:,:,k)'*Qux + Qux'*K_seq(:,:,k);
        Vxx = 0.5*(Vxx+Vxx');  % enforce symmetry
    end
    
    %--- Forward Pass with Line Search ---
    alpha = 1.0;
    cost_new = Inf;
    u_seq_new = u_seq;
    x_new = zeros(4, N+1);
    while alpha > 1e-4
        x_new(:,1) = x0;
        cost_new = 0;
        for k = 1:N
            % Use feedforward and feedback control corrections
            du = alpha * k_seq(k) + K_seq(:,:,k) * (x_new(:,k) - x_traj(:,k));
            u_new = u_seq(k) + du;
            u_seq_new(k) = u_new;
            x_new(:,k+1) = rk4(@(x,u) dynamics(x,u,a,b,c,m_p,g,l), x_new(:,k), u_new, dt);
            dx = x_new(:,k) - x_target;
            cost_new = cost_new + dx'*Q*dx + u_new'*R*u_new;
        end
        dx = x_new(:,end) - x_target;
        cost_new = cost_new + dx'*Qf*dx;
        
        if cost_new < cost
            break;
        else
            alpha = alpha * 0.5;
        end
    end
    
    if abs(cost - cost_new) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    
    u_seq = u_seq_new;
end

%% Plotting Results: Only Wheel Position and Pendulum Angle
t = 0:dt:T;
figure;
subplot(2,1,1);
plot(t, x_new(1,:), 'LineWidth',1.5);
ylabel('Wheel Position x (m)');
title('Closed-Loop Response');
grid on;

subplot(2,1,2);
plot(t, x_new(3,:), 'LineWidth',1.5);
ylabel('Pendulum Angle \theta (rad)');
xlabel('Time (s)');
grid on;

%% --- Dynamics Function ---
function xdot = dynamics(x, u, a, b, c, m_p, g, l)
    % Nonlinear dynamics for the Segway-like system.
    % x: state vector [x; xdot; theta; theta_dot]
    % u: control input F
    theta = x(3);
    theta_dot = x(4);
    Delta = a*c - b^2*cos(theta)^2;
    
    xdot = zeros(4,1);
    xdot(1) = x(2);
    % Horizontal acceleration
    xdot(2) = ( c*(u + b*sin(theta)*theta_dot^2) - b*cos(theta)*(m_p*g*l*sin(theta)) ) / Delta;
    % Pendulum kinematics
    xdot(3) = theta_dot;
    % Angular acceleration
    xdot(4) = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*(u + b*sin(theta)*theta_dot^2) ) / Delta;
end

%% --- RK4 Integrator Function ---
function x_next = rk4(f, x, u, dt)
    % Fourth-order Runge-Kutta integrator for one time step.
    k1 = f(x, u);
    k2 = f(x + dt/2 * k1, u);
    k3 = f(x + dt/2 * k2, u);
    k4 = f(x + dt * k3, u);
    x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end