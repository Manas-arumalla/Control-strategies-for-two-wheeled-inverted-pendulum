clear; clc; close all;

%% System Parameters
m_w = 0.432;      
m_p = 5.0;        
I_w = 1/2 * m_w * (0.0726)^2;    
I_p = 5 * (0.4)^2;               
l   = 0.4;        
r   = 0.0726;      
g   = 9.81;        

a = 2*m_w + m_p + 2*I_w/(r^2);
b = m_p * l;
c = m_p * l^2 + I_p;

%% Simulation Settings
dt = 0.01;
T  = 5;
N  = round(T/dt);

x_target = [0; 0; 0; 0];
x0 = [0; 0; 0.1; 0];

%% GA Optimization
% Decision variables: 4 Qs, 4 Qfs, 1 R
nvars = 9; 
lb = [1 1 1 1  10 10 10 10  0.0001]; 
ub = [500 500 5000 500  5000 5000 50000 5000  1];

fitnessFcn = @(params) cost_function_ga(params, x0, x_target, a,b,c,m_p,g,l,N,dt);
options = optimoptions('ga','MaxGenerations',150,'Display','iter','UseParallel',false);
[opt_params,~] = ga(fitnessFcn,nvars,[],[],[],[],lb,ub,[],options);

Q  = diag(opt_params(1:4));
Qf = diag(opt_params(5:8));
R  = opt_params(9);

%% Run iLQR with Optimal Parameters
[u_seq, x_new] = run_ilqr(x0, x_target, Q, Qf, R, a,b,c,m_p,g,l,N,dt);

%% Plotting Final Results
t = 0:dt:T;
figure;
subplot(2,1,1);
plot(t, x_new(1,:), 'LineWidth',1.5);
ylabel('Wheel Position x (m)');
title('Closed-Loop Response with Optimal GA-Tuned iLQR');
grid on;

subplot(2,1,2);
plot(t, x_new(3,:), 'LineWidth',1.5);
ylabel('Pendulum Angle \theta (rad)');
xlabel('Time (s)');
grid on;

%% --- Cost Function for GA ---
function total_cost = cost_function_ga(params, x0, x_target, a,b,c,m_p,g,l,N,dt)
    Q  = diag(params(1:4));
    Qf = diag(params(5:8));
    R  = params(9);

    [u_seq, x_traj] = run_ilqr(x0, x_target, Q, Qf, R, a,b,c,m_p,g,l,N,dt);

    total_cost = 0;
    for k = 1:N
        dx = x_traj(:,k) - x_target;
        total_cost = total_cost + dx'*Q*dx + u_seq(k)'*R*u_seq(k);
    end
    dx = x_traj(:,end) - x_target;
    total_cost = total_cost + dx'*Qf*dx;
end

%% --- iLQR Function ---
function [u_seq, x_new] = run_ilqr(x0, x_target, Q, Qf, R, a,b,c,m_p,g,l,N,dt)
    max_iter = 100;
    tol = 1e-4;
    u_seq = zeros(1,N);

    for iter = 1:max_iter
        x_traj = zeros(4,N+1);
        x_traj(:,1) = x0;
        for k = 1:N
            x_traj(:,k+1) = rk4(@(x,u) dynamics(x,u,a,b,c,m_p,g,l), x_traj(:,k), u_seq(k), dt);
        end

        A_seq = zeros(4,4,N);
        B_seq = zeros(4,1,N);
        eps_fd = 1e-5;
        for k = 1:N
            xk = x_traj(:,k); uk = u_seq(k);
            A = zeros(4,4);
            for i = 1:4
                dx = zeros(4,1); dx(i)=eps_fd;
                A(:,i) = (dynamics(xk+dx,uk,a,b,c,m_p,g,l)-dynamics(xk-dx,uk,a,b,c,m_p,g,l))/(2*eps_fd);
            end
            du = eps_fd;
            B = (dynamics(xk,uk+du,a,b,c,m_p,g,l)-dynamics(xk,uk-du,a,b,c,m_p,g,l))/(2*du);

            A_seq(:,:,k) = eye(4) + A*dt;
            B_seq(:,:,k) = B*dt;
        end

        Vx = Qf*(x_traj(:,end) - x_target);
        Vxx = Qf;
        k_seq = zeros(1,N);
        K_seq = zeros(1,4,N);
        reg = 1e-6;

        for k = N:-1:1
            Qx = Q*(x_traj(:,k)-x_target) + A_seq(:,:,k)'*Vx;
            Qu = R*u_seq(k) + B_seq(:,:,k)'*Vx;
            Qxx = Q + A_seq(:,:,k)'*Vxx*A_seq(:,:,k);
            Quu = R + B_seq(:,:,k)'*Vxx*B_seq(:,:,k) + reg*eye(1);
            Qux = B_seq(:,:,k)'*Vxx*A_seq(:,:,k);

            invQuu = inv(Quu);
            k_seq(k) = -invQuu*Qu;
            K_seq(:,:,k) = -invQuu*Qux;

            Vx = Qx + K_seq(:,:,k)'*Quu*k_seq(k) + K_seq(:,:,k)'*Qu + Qux'*k_seq(k);
            Vxx = Qxx + K_seq(:,:,k)'*Quu*K_seq(:,:,k) + K_seq(:,:,k)'*Qux + Qux'*K_seq(:,:,k);
            Vxx = 0.5*(Vxx+Vxx');
        end

        alpha = 1.0;
        u_seq_new = u_seq;
        x_new = zeros(4,N+1);
        while alpha > 1e-4
            x_new(:,1) = x0;
            for k = 1:N
                du = alpha*k_seq(k) + K_seq(:,:,k)*(x_new(:,k)-x_traj(:,k));
                u_new = u_seq(k) + du;
                u_seq_new(k) = u_new;
                x_new(:,k+1) = rk4(@(x,u) dynamics(x,u,a,b,c,m_p,g,l), x_new(:,k), u_new, dt);
            end
            break;
        end

        if norm(u_seq_new - u_seq) < tol
            break;
        end
        u_seq = u_seq_new;
    end
end

%% --- Dynamics ---
function xdot = dynamics(x, u, a, b, c, m_p, g, l)
    theta = x(3);
    theta_dot = x(4);
    Delta = a*c - b^2*cos(theta)^2;

    xdot = zeros(4,1);
    xdot(1) = x(2);
    xdot(2) = ( c*(u + b*sin(theta)*theta_dot^2) - b*cos(theta)*(m_p*g*l*sin(theta)) ) / Delta;
    xdot(3) = theta_dot;
    xdot(4) = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*(u + b*sin(theta)*theta_dot^2) ) / Delta;
end

%% --- RK4 Integrator ---
function x_next = rk4(f, x, u, dt)
    k1 = f(x,u);
    k2 = f(x + dt/2 * k1, u);
    k3 = f(x + dt/2 * k2, u);
    k4 = f(x + dt * k3, u);
    x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end
