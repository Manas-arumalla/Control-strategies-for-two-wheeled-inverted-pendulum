function inverted_pendulum_NDI_SMC
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
    
    % Sliding mode control gains
    % For the cart (horizontal motion)
    lambda_x = 2.0;
    kx = 5.0;
    % For the pendulum (angle stabilization)
    lambda_theta = 10.0;
    k_theta = 20.0;
    
    % Simulation settings
    tspan = [0 10];             % simulation time (sec)
    % Initial conditions: [x; x_dot; theta; theta_dot]
    x0 = [0; 0; 0.2; 0];         % small initial angular deviation
    
    % Run simulation using ODE45
    [t, x] = ode45(@(t,x) dynamics_SMC(t, x, a, b, c, mp, l, g, lambda_x, kx, lambda_theta, k_theta), tspan, x0);
    
    % Plot results
    figure;
    subplot(2,1,1);
    plot(t, x(:,1), 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Cart Position (m)');
    title('Cart Position vs Time'); grid on;
    
    subplot(2,1,2);
    plot(t, x(:,3), 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
    title('Pendulum Angle vs Time'); grid on;
end

function dx = dynamics_SMC(~, x, a, b, c, mp, l, g, lambda_x, kx, lambda_theta, k_theta)
    % State variables
    % x(1) = x, x(2) = x_dot, x(3) = theta, x(4) = theta_dot
    pos       = x(1);
    vel       = x(2);
    theta     = x(3);
    theta_dot = x(4);
    
    % Define sliding surfaces for the cart and pendulum:
    s_x = vel + lambda_x * pos;
    s_theta = theta_dot + lambda_theta * theta;
    
    % Sliding mode control law:
    % Desired horizontal acceleration from SMC for cart position
    x_ddot_des = -lambda_x * vel - kx * tanh(s_x);
    % Desired virtual control for pendulum stabilization
    v = -lambda_theta * theta_dot - k_theta * tanh(s_theta);
    
    % Inertia matrix M(q)
    M = [ a,           b*cos(theta);
          b*cos(theta),    c ];
      
    % Nonlinear terms vector C(q, q_dot)
    % Equations:
    % Eq1: a*x_ddot + b*cos(theta)*theta_ddot - b*sin(theta)*theta_dot^2 = F
    % Eq2: b*cos(theta)*x_ddot + c*theta_ddot - mp*g*l*sin(theta) = 0
    C = [ -b*sin(theta)*theta_dot^2;
           mp*g*l*sin(theta) ];
    
    % Desired accelerations vector for [x; theta]
    desired_acc = [x_ddot_des; v];
    
    % Compute the term M * desired_acc and include nonlinear terms
    U = M * desired_acc + C;
    
    % The control input for the wheels is F (the first element of U)
    F = U(1);
    
    % Compute the actual accelerations using the dynamic inversion:
    q_ddot = M \ ([F; v] - C);
    x_ddot = q_ddot(1);
    theta_ddot = q_ddot(2);
    
    % Assemble state derivative vector
    dx = zeros(4,1);
    dx(1) = vel;
    dx(2) = x_ddot;
    dx(3) = theta_dot;
    dx(4) = theta_ddot;
end
