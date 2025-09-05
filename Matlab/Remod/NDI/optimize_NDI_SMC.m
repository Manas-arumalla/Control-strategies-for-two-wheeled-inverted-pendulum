function optimize_NDI_SMC
    % Optimize SMC controller gains for an inverted pendulum using GA.
    % The gains to be optimized are:
    % p = [lambda_x, kx, lambda_theta, k_theta]
    
    % Define lower and upper bounds for the gains.
    % Adjust these bounds as needed.
    lb = [0.1, 0.1, 0.1, 0.1];
    ub = [5,   50,  20,  50];
    
    % GA options
    options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
    costFun = @(p) smcCostFunction(p);
    
    % Run GA optimization
    [p_opt, fval] = ga(costFun, 4, [], [], [], [], lb, ub, [], options);
    
    fprintf('\nOptimized SMC Gains:\n');
    fprintf('lambda_x = %.3f, kx = %.3f, lambda_theta = %.3f, k_theta = %.3f\n', ...
            p_opt(1), p_opt(2), p_opt(3), p_opt(4));
    fprintf('Final Cost: %.3f\n', fval);
    
    % Simulate the system with the optimized gains
    tspan = [0 10];
    x0 = [0; 0; 0.1; 0];  % Initial condition: small angular deviation
    [t, x] = ode45(@(t,x) dynamics_SMC_optim(t, x, p_opt), tspan, x0);
    
    % Plot the results
    figure;
    subplot(2,1,1);
    plot(t, x(:,1), 'b','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Cart Position (m)');
    title('Optimized SMC: Cart Position vs Time'); grid on;
    
    subplot(2,1,2);
    plot(t, x(:,3), 'r','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
    title('Optimized SMC: Pendulum Angle vs Time'); grid on;

    % Desired reference is zero for both cart position and pendulum angle
    desired_reference = 0;
    tolerance = 0.01;  % Tolerance for settling time
    
    % Responses
    position_response = x(:,1);  % Cart/Wheel Position
    angle_response = x(:,3);     % Pendulum Angle
    
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

end

%% Cost Function for SMC Controller Optimization
function J = smcCostFunction(p)
    % p = [lambda_x, kx, lambda_theta, k_theta]
    % Plant parameters (provided):
    m_w = 0.432;         % mass of each wheel [kg]
    mp  = 5.0;           % mass of the pendulum [kg]
    Iw  = 1/2 * (0.432) * (0.0726)^2;   % wheel inertia [kg*m^2]
    Ip  = 5 * (0.4)^2;    % pendulum inertia [kg*m^2]
    l   = 0.4;          % pendulum length [m]
    r   = 0.0726;       % wheel radius [m]
    g   = 9.81;         % gravity [m/s^2]
    
    % Derived parameters:
    a = 2*m_w + mp + 2*Iw/(r^2);
    b = mp * l;
    c = mp * l^2 + Ip;
    
    % Simulation settings
    tspan = [0 10];
    x0 = [0; 0; 0.1; 0];  % small initial angular deviation
    
    % Simulate the closed-loop system using the SMC dynamics
    [t, x] = ode45(@(t,x) dynamics_SMC_optim(t, x, p), tspan, x0);
    
    % Desired references: zero cart position and zero pendulum angle
    ref = [0; 0];
    % Extract outputs: cart position x and pendulum angle theta
    y = [x(:,1)'; x(:,3)'];
    
    % Compute tracking error (integrated squared error)
    err = ref - y;
    J = trapz(t, err(1,:).^2 + err(2,:).^2);
    
    % Add a penalty if the system does not settle near the reference
    if abs(err(1,end)) > 0.05 || abs(err(2,end)) > 0.05
        J = J + 1e3;
    end
end

%% Dynamics Function for Inverted Pendulum with SMC (for GA optimization)
function dx = dynamics_SMC_optim(~, x, p)
    % p = [lambda_x, kx, lambda_theta, k_theta]
    lambda_x = p(1);
    kx = p(2);
    lambda_theta = p(3);
    k_theta = p(4);
    
    % Plant parameters (provided)
    m_w = 0.432;         % mass of each wheel [kg]
    mp  = 5.0;           % mass of the pendulum [kg]
    Iw  = 1/2 * (0.432) * (0.0726)^2;   % moment of inertia for each wheel [kg*m^2]
    Ip  = 5 * (0.4)^2;    % moment of inertia of the pendulum [kg*m^2]
    l   = 0.4;          % Length of the pendulum [m]
    r   = 0.0726;       % wheel radius [m]
    g   = 9.81;         % acceleration due to gravity [m/s^2]
    
    % Derived parameters
    a = 2*m_w + mp + 2*Iw/(r^2);  % effective horizontal inertia
    b = mp * l;
    c = mp * l^2 + Ip;
    
    % State variables:
    % x(1) = cart position, x(2) = cart velocity,
    % x(3) = pendulum angle, x(4) = pendulum angular velocity
    pos       = x(1);
    vel       = x(2);
    theta     = x(3);
    theta_dot = x(4);
    
    % Define sliding surfaces for the cart and pendulum
    s_x = vel + lambda_x * pos;
    s_theta = theta_dot + lambda_theta * theta;
    
    % Sliding mode control law (using tanh for smoother control)
    x_ddot_des = -lambda_x * vel - kx * tanh(s_x);
    v = -lambda_theta * theta_dot - k_theta * tanh(s_theta);
    
    % Inertia matrix M(q)
    M = [ a,           b*cos(theta);
          b*cos(theta),    c ];
      
    % Nonlinear terms vector C(q, q_dot)
    C = [ -b*sin(theta)*theta_dot^2;
           mp*g*l*sin(theta) ];
    
    % Desired accelerations vector for [x; theta]
    desired_acc = [x_ddot_des; v];
    
    % Dynamic inversion: Compute U = M*desired_acc + C
    U = M * desired_acc + C;
    F = U(1);  % control input for the wheels
    
    % Compute the actual accelerations via inversion
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
