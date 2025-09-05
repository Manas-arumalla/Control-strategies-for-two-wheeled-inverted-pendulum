function optimize_Sontag
    % Optimize Sontag feedback controller parameters for a Segway model using GA.
    % Decision vector: p = [w1, w2, w3, w4, w5, F_max]

    % Lower and upper bounds for the parameters:
    % w1, w2: weights on x and xdot; w3, w4: weights on theta and theta_dot;
    % w5: cross coupling weight; F_max: control saturation limit.
    lb = [0.1, 0.1, 0.1, 0.1, 0, 10];
    ub = [10, 10, 20, 10, 5, 100];

    options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50, 'MaxGenerations', 200);
    costFun = @(p) sontagCostFunction(p);
    
    [p_opt, fval] = ga(costFun, 6, [], [], [], [], lb, ub, [], options);
    
    fprintf('\nOptimized Sontag Parameters:\n');
    fprintf('w1 = %.3f, w2 = %.3f, w3 = %.3f, w4 = %.3f, w5 = %.3f, F_max = %.3f\n', ...
            p_opt(1), p_opt(2), p_opt(3), p_opt(4), p_opt(5), p_opt(6));
    
    % Simulation with the optimized parameters:
    tspan = [0 10];
    x0 = [0; 0; 0.1; 0];  % initial state: [position; velocity; angle; angular velocity]
    
    % Segway system parameters (example values)
    m_w = 1.0;      % mass of each wheel [kg]
    I_W = 0.02;     % moment of inertia of each wheel [kg*m^2]
    m_p = 5.0;      % mass of pendulum [kg]
    I_p = 0.2;      % moment of inertia of pendulum [kg*m^2]
    l   = 0.5;      % distance from axle to pendulum center of mass [m]
    r   = 0.1;      % wheel radius [m]
    g   = 9.81;     % gravitational acceleration [m/s^2]
    
    % Derived constants:
    a = 2*m_w + m_p + 2*I_W/(r^2);
    b = m_p * l;
    c = m_p * l^2 + I_p;
    
    [t, X] = ode45(@(t,x) segwayDynamics(t, x, a, b, c, m_p, g, l, ...
                           p_opt(1), p_opt(2), p_opt(3), p_opt(4), p_opt(5), p_opt(6)), tspan, x0);
    
    % Plotting the results:
    figure;
    subplot(2,1,1);
    plot(t, X(:,3), 'r', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('\theta (rad)');
    title('Pendulum Angle vs Time (Optimized Sontag)');
    grid on;
    
    subplot(2,1,2);
    plot(t, X(:,1), 'b', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('x (m)');
    title('Axle Position vs Time (Optimized Sontag)');
    grid on;
end

%% Cost Function for Sontag Feedback Controller Optimization
function J = sontagCostFunction(p)
    % p = [w1, w2, w3, w4, w5, F_max]
    % Segway system parameters (same as in simulation)
    m_w = 1.0;      % mass of each wheel [kg]
    I_W = 0.02;     % moment of inertia of each wheel [kg*m^2]
    m_p = 5.0;      % mass of pendulum [kg]
    I_p = 0.2;      % moment of inertia of pendulum [kg*m^2]
    l   = 0.5;      % distance from axle to pendulum center of mass [m]
    r   = 0.1;      % wheel radius [m]
    g   = 9.81;     % gravitational acceleration [m/s^2]
    
    % Derived constants:
    a = 2*m_w + m_p + 2*I_W/(r^2);
    b = m_p * l;
    c = m_p * l^2 + I_p;
    
    % Check the positive definiteness condition for the Lyapunov function:
    % To ensure V is positive definite, we require: w1*w3 > w5^2.
    if p(1)*p(3) <= p(5)^2
        J = 1e6;  % assign a large penalty if the condition is not met
        return;
    end
    
    % Simulation settings:
    tspan = [0 10];
    x0 = [0; 0; 0.1; 0];  % small initial deviation
    
    % Simulate the closed-loop system:
    [t, x] = ode45(@(t,x) segwayDynamics(t, x, a, b, c, m_p, g, l, ...
                          p(1), p(2), p(3), p(4), p(5), p(6)), tspan, x0);
    
    % Define the tracking error: we desire x (cart position) and theta (pendulum angle)
    % to converge to zero.
    err = x(:,1).^2 + x(:,3).^2;
    J = trapz(t, err);
    
    % Add an extra penalty if the system does not settle sufficiently (error above threshold)
    if (abs(x(end,1)) > 0.05) || (abs(x(end,3)) > 0.05)
        J = J + 1e4;
    end
end

%% Dynamics Function with Sontag's Feedback Control
function dx = segwayDynamics(~, x, a, b, c, m_p, g, l, w1, w2, w3, w4, w5, F_max)
    % x = [x; xdot; theta; theta_dot]
    x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4);
    
    % Pre-calculate trigonometric terms and Delta:
    cosTheta = cos(x3);
    sinTheta = sin(x3);
    Delta = a*c - b^2*cosTheta^2;
    
    % Drift terms (system dynamics without control):
    f2 = (c*b*sinTheta*x4^2 - b*m_p*g*l*sinTheta*cosTheta) / Delta;
    f4 = (-b^2*sinTheta*cosTheta*x4^2 + a*m_p*g*l*sinTheta) / Delta;
    
    % Control influence terms:
    g2 = c / Delta;
    g4 = -b*cosTheta / Delta;
    
    % Assemble the drift vector and control input influence:
    f = [ x2;
          f2;
          x4;
          f4 ];
    g_vec = [ 0;
              g2;
              0;
              g4 ];
    
    % Candidate Lyapunov function:
    % V = 0.5*(w1*x1^2 + w2*x2^2 + w3*x3^2 + w4*x4^2) + w5*x1*x3
    dVdx = [ w1*x1 + w5*x3;
             w2*x2;
             w3*x3 + w5*x1;
             w4*x4 ];
    
    % Compute the Lie derivatives:
    LfV = dVdx' * f;
    LgV = dVdx' * g_vec;
    
    % Sontag's universal formula (with a regularization parameter to avoid division by zero):
    eps = 1e-4;
    if abs(LgV) > eps
        F_unsat = - ( LfV + sqrt(LfV^2 + LgV^4) ) / LgV;
    else
        F_unsat = 0;
    end
    
    % Saturate the control input:
    F = max(min(F_unsat, F_max), -F_max);
    
    % Full state dynamics with control:
    dx = f + g_vec * F;
end
