function optimize_NDI
    % Optimize NDI controller gains for an inverted pendulum using GA.
    
    % Define lower and upper bounds for the gains:
    % Kpx: proportional gain for cart position, Kdx: derivative gain for cart velocity,
    % Kp_theta: proportional gain for pendulum angle, Kd_theta: derivative gain for pendulum angular velocity.
    lb = [0.1, 0.1, 1, 1];
    ub = [10, 20, 200, 50];
    
    options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
    costFun = @(p) ndiCostFunction(p);
    
    [p_opt, ~] = ga(costFun, 4, [], [], [], [], lb, ub, [], options);
    
    fprintf('\nOptimized NDI Gains:\n');
    fprintf('Kpx = %.3f, Kdx = %.3f, Kp_theta = %.3f, Kd_theta = %.3f\n', p_opt(1), p_opt(2), p_opt(3), p_opt(4));
    
    % Simulate the system with the optimized gains
    tspan = [0 10];
    x0 = [0; 0; 0.2; 0]; % initial state (small angular deviation)
    [t, x] = ode45(@(t,x) dynamics(t, x, p_opt), tspan, x0);
    
    figure;
    subplot(2,1,1);
    plot(t, x(:,1), 'b','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Cart Position (m)');
    title('Optimized NDI: Cart Position vs Time'); grid on;
    
    subplot(2,1,2);
    plot(t, x(:,3), 'r','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
    title('Optimized NDI: Pendulum Angle vs Time'); grid on;

end

%% Cost Function for NDI Controller Optimization
function J = ndiCostFunction(p)
    % p = [Kpx, Kdx, Kp_theta, Kd_theta]
    % Use the provided plant parameters.
    m_w = 0.432;      % mass of each wheel [kg]
    mp = 5.0;        % mass of the pendulum [kg]
    Iw = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia of each wheel [kg*m^2]
    Ip = 5 * (0.4)^2;                    % moment of inertia of the pendulum [kg*m^2]
    l   = 0.4;      % Length of the pendulum [m]
    r   = 0.0726;   % wheel radius [m]
    g   = 9.81;     % acceleration due to gravity [m/s^2]
    
    % Derived parameters (as given in your script)
    a = 2*m_w + mp + 2*Iw/(r^2);  % effective horizontal inertia
    b = mp * l;
    c = mp * l^2 + Ip;
    
    % Simulation settings
    tspan = [0 10];
    x0 = [0; 0; 0.1; 0];  % initial condition with small angular deviation
    
    % Simulate the closed-loop system
    [t, x] = ode45(@(t,x) dynamics(t, x, p), tspan, x0);
    
    % Define desired reference (zero for both cart position and pendulum angle)
    ref = [0; 0];
    
    % Compute output: x(1) is cart position and x(3) is pendulum angle.
    y = [x(:,1)'; x(:,3)'];
    
    % Tracking error (we use squared error integrated over time)
    err = ref - y;
    J = trapz(t, err(1,:).^2 + err(2,:).^2);
    
    % Optionally add a penalty if the system does not settle (if error remains high near end)
    if abs(err(1,end)) > 0.05 || abs(err(2,end)) > 0.05
        J = J + 1e3;
    end
end

%% Dynamics Function for the Inverted Pendulum with NDI Controller
function dx = dynamics(~, x, p)
    % p = [Kpx, Kdx, Kp_theta, Kd_theta]
    Kpx = p(1);
    Kdx = p(2);
    Kp_theta = p(3);
    Kd_theta = p(4);
    
    % System parameters (same as in your NDI script)
    mw = 0.432;       % mass of one wheel (kg)
    mp = 5.0;       % mass of the pendulum (kg)
    Iw = 1/2 * (0.432) * (0.0726)^2;     % moment of inertia for one wheel (kg.m^2)
    Ip = 5 * (0.4)^2;      % moment of inertia of the pendulum about its COM (kg.m^2)
    r = 0.0726;        % wheel radius (m)
    l = 0.4;        % distance from axle to pendulum COM (m)
    g = 9.81;       % gravitational acceleration (m/s^2)
    
    % Derived parameters:
    a = 2*mw + mp + 2*Iw/(r^2);
    b = mp * l;
    c = mp * l^2 + Ip;
    
    % Extract states:
    pos = x(1);          % cart position
    vel = x(2);          % cart velocity
    theta = x(3);        % pendulum angle
    theta_dot = x(4);    % pendulum angular velocity
    
    % Desired dynamics for cart:
    x_ddot_des = -Kpx * pos - Kdx * vel;
    % Desired dynamics for pendulum (stabilization):
    v = -Kp_theta * theta - Kd_theta * theta_dot;
    
    % Inertia matrix M(q)
    M = [ a,              b*cos(theta);
          b*cos(theta),   c ];
    
    % Nonlinear terms vector:
    % From the equations:
    % a*x_ddot + b*cos(theta)*theta_ddot - b*sin(theta)*theta_dot^2 = F
    % b*cos(theta)*x_ddot + c*theta_ddot - mp*g*l*sin(theta) = 0
    C_vec = [ -b*sin(theta)*theta_dot^2;
              mp*g*l*sin(theta) ];
    
    % Desired acceleration vector for [x; theta]:
    desired_acc = [x_ddot_des; v];
    
    % Compute M*desired_acc
    M_desired = M * desired_acc;
    
    % Dynamic inversion:
    U = M_desired + C_vec;
    F = U(1);  % control input for the wheels
    
    % Now, compute the actual accelerations:
    q_ddot = M \ ([F; v] - C_vec);
    x_ddot = q_ddot(1);
    theta_ddot = q_ddot(2);
    
    dx = zeros(4,1);
    dx(1) = vel;
    dx(2) = x_ddot;
    dx(3) = theta_dot;
    dx(4) = theta_ddot;
end
