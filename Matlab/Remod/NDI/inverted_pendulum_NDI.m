%% Inverted Pendulum Control using Nonlinear Dynamic Inversion (NDI)
% This script simulates a wheeled inverted pendulum controlled via NDI.
% The controller is designed to stabilize the pendulum (theta) and the
% horizontal position (x) to zero.

function inverted_pendulum_NDI
    % System parameters
    % Use the provided plant parameters.
    mw = 0.432;      % mass of each wheel [kg]
    mp = 5.0;        % mass of the pendulum [kg]
    Iw = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
    Ip = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
    l   = 0.4;      % Length of the pendulum [m]
    r   = 0.0726;   % wheel radius [m]
    g   = 9.81;     % acceleration due to gravity [m/s^2]
    
    % Derived parameters
    a = 2*mw + mp + 2*Iw/(r^2);  % effective horizontal inertia
    b = mp * l;
    c = mp * l^2 + Ip;
    
    % Controller gains
    % Gains for horizontal position control
    Kpx = 1.0;
    Kdx = 2.0;
    % Gains for pendulum angle control (stabilization)
    Kp_theta = 50;
    Kd_theta = 10;
    
    % Simulation settings
    tspan = [0 10]; % simulation time (sec)
    % Initial conditions: [x; x_dot; theta; theta_dot]
    x0 = [0; 0; 0.2; 0];  % start with a small angular deviation
    
    % Run simulation using ODE45
    [t, x] = ode45(@(t,x) dynamics(t, x, a, b, c, mp, l, g, Kpx, Kdx, Kp_theta, Kd_theta), tspan, x0);
    
    % Plot results
    figure;
    subplot(2,1,1);
    plot(t, x(:,1), 'LineWidth', 1.5);
    xlabel('Time (s)'); ylabel('cart Position (m)');
    title('cart Position vs Time'); grid on;
    
    subplot(2,1,2);
    plot(t, x(:,3), 'LineWidth', 1.5);
    xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
    title('Pendulum Angle vs Time'); grid on;
end

function dx = dynamics(~, x, a, b, c, mp, l, g, Kpx, Kdx, Kp_theta, Kd_theta)
    % State variables
    % x(1) = x, x(2) = x_dot, x(3) = theta, x(4) = theta_dot
    
    % Extract states
    pos    = x(1);
    vel    = x(2);
    theta  = x(3);
    theta_dot = x(4);
    
    % Desired dynamics:
    % For horizontal motion, use PD control to compute desired acceleration:
    x_ddot_des = -Kpx * pos - Kdx * vel;
    
    % For pendulum, desire theta -> 0:
    v = -Kp_theta * theta - Kd_theta * theta_dot;
    
    % Inertia matrix M(q)
    M = [ a,          b*cos(theta);
          b*cos(theta),   c ];
      
    % Nonlinear terms vector C(q, q_dot)
    % From the equations:
    % Eq1: a*x_ddot + b*cos(theta)*theta_ddot - b*sin(theta)*theta_dot^2 = F
    % Eq2: b*cos(theta)*x_ddot + c*theta_ddot - mp*g*l*sin(theta) = 0
    % Rearranged: 
    % Let C = [ -b*sin(theta)*theta_dot^2;
    %           -(-mp*g*l*sin(theta)) ]
    % Here we write:
    C = [ -b*sin(theta)*theta_dot^2;
           mp*g*l*sin(theta) ];
    
    % Desired accelerations vector for [x; theta]:
    desired_acc = [x_ddot_des; v];
    
    % Compute the term M * desired_acc
    M_desired = M * desired_acc;
    
    % Dynamic inversion: choose control input vector U = [F; v] such that:
    % U = M*desired_acc + C, which gives the desired acceleration when inverted.
    U = M_desired + C;
    
    % The control input for the wheels is F (the first element)
    F = U(1);
    
    % Now, compute the actual accelerations achieved by the system:
    q_ddot = M \ ( [F; v] - C );
    x_ddot = q_ddot(1);
    theta_ddot = q_ddot(2);
    
    % Assemble state derivative vector
    dx = zeros(4,1);
    dx(1) = vel;
    dx(2) = x_ddot;
    dx(3) = theta_dot;
    dx(4) = theta_ddot;
end

