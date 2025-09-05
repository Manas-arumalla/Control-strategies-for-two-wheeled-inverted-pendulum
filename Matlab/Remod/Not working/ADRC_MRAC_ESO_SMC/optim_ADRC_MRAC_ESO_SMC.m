function optimize_ADRC_SMC
    % Optimize ADRC-ESO/SMC controller parameters via GA.
    % Decision vector: p = [w0, lambda_x, lambda_theta, alpha_weight, K0, eta, epsilon, gamma]
    
    %% Set Bounds for Decision Variables
    % w0: observer bandwidth [20, 100]
    % lambda_x: SMC slope for cart [1, 10]
    % lambda_theta: SMC slope for pendulum [5, 20]
    % alpha_weight: weighting factor [0, 1]
    % K0: initial adaptive gain [10, 100]
    % eta: robust gain [1, 10]
    % epsilon: smoothing parameter [0.001, 0.05]
    % gamma: adaptation rate [1, 20]
    lb = [20, 1, 5, 0, 10, 1, 0.001, 1];
    ub = [100, 10, 20, 1, 100, 10, 0.05, 20];
    
    options = optimoptions('ga','Display','iter','PopulationSize',50,'MaxGenerations',200);
    costFun = @(p) costFunctionADRC_SMC(p);
    
    [p_opt, fval] = ga(costFun, 8, [], [], [], [], lb, ub, [], []);
    
    fprintf('\nOptimized Parameters:\n');
    fprintf('w0 = %.2f\n', p_opt(1));
    fprintf('lambda_x = %.2f\n', p_opt(2));
    fprintf('lambda_theta = %.2f\n', p_opt(3));
    fprintf('alpha_weight = %.2f\n', p_opt(4));
    fprintf('K0 = %.2f\n', p_opt(5));
    fprintf('eta = %.2f\n', p_opt(6));
    fprintf('epsilon = %.4f\n', p_opt(7));
    fprintf('gamma = %.2f\n', p_opt(8));
    
    % Final simulation with optimized parameters:
    simulateController(p_opt);
end

%% Cost Function for ADRC-ESO/SMC Controller
function J = costFunctionADRC_SMC(p)
    % p = [w0, lambda_x, lambda_theta, alpha_weight, K0, eta, epsilon, gamma]
    % Fixed system parameters:
    m_w = 0.432;
    m_p = 5.0;
    I_w = 0.5 * (0.432) * (0.0726)^2;
    I_p = 5 * (0.4)^2;
    l = 0.4;
    r = 0.0726;
    g = 9.81;
    
    % Derived parameters:
    a = 2*m_w + m_p + 2*I_w/(r^2);
    b = m_p * l;
    c = m_p * l^2 + I_p;
    d = m_p * g * l;
    Delta = a*c - b^2;
    b0 = -b / Delta;
    
    % Extract decision variables:
    w0 = p(1);
    lambda_x = p(2);
    lambda_theta = p(3);
    alpha_weight = p(4);
    K_val = p(5);
    eta = p(6);
    epsilon = p(7);
    gamma = p(8);
    
    % Compute ESO gains:
    L1 = 3*w0;
    L2 = 3*w0^2;
    L3 = w0^3;
    
    % Simulation settings:
    dt = 0.001;
    T_final = 10;
    t = 0:dt:T_final;
    N = length(t);
    
    % Initial conditions:
    x = zeros(4, N);
    x(:,1) = [0; 0; 0.1; 0];
    
    % Preallocate control and adaptive gain histories:
    u_total = zeros(1, N);
    K_hist = zeros(1, N);
    K_hist(1) = K_val;
    
    % ESO initializations:
    z_x = x(1,1);  z_xdot = x(2,1);  z_dist_x = 0;
    z_theta = x(3,1);  z_thetadot = x(4,1);  z_dist_theta = 0;
    
    % Initialize cost accumulator:
    J = 0;
    
    % Run simulation:
    for k = 1:N-1
        x_meas = x(:,k);
        cart_pos = x_meas(1);
        cart_vel = x_meas(2);
        theta = x_meas(3);
        theta_dot = x_meas(4);
        
        % ESO Updates:
        e_x_obs = cart_pos - z_x;
        z_x = z_x + dt*(z_xdot + L1*e_x_obs);
        z_xdot = z_xdot + dt*(z_dist_x + L2*e_x_obs);
        z_dist_x = z_dist_x + dt*(L3*e_x_obs);
        
        e_theta_obs = theta - z_theta;
        z_theta = z_theta + dt*(z_thetadot + L1*e_theta_obs);
        z_thetadot = z_thetadot + dt*(z_dist_theta + L2*e_theta_obs);
        z_dist_theta = z_dist_theta + dt*(L3*e_theta_obs);
        
        % Sliding surfaces:
        e_x = 0 - z_x;
        e_xdot = 0 - z_xdot;
        e_theta = 0 - z_theta;
        e_thetadot = 0 - z_thetadot;
        s_x = e_x + lambda_x * e_xdot;
        s_theta = e_theta + lambda_theta * e_thetadot;
        s = alpha_weight * s_x + (1 - alpha_weight) * s_theta;
        
        % Control law:
        u_nom = - K_val * s;
        u_robust = - eta * tanh(s/epsilon);
        u_ASMC = u_nom + u_robust;
        u = (u_ASMC - (z_dist_x + z_dist_theta)) / b0;
        u = max(min(u, 10), -10);
        u_total(k) = u;
        
        % Adaptive gain update:
        K_val = K_val + gamma * s^2 * dt;
        K_hist(k+1) = K_val;
        
        % Plant Dynamics (nonlinear model):
        denom = a*c - b^2 * cos(theta)^2;
        ddx = ( c*( u + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*( m_p*g*l*sin(theta) ) ) / denom;
        ddtheta = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*( u + b*sin(theta)*theta_dot^2 ) ) / denom;
        dx = [ cart_vel; ddx; theta_dot; ddtheta ];
        x(:,k+1) = x(:,k) + dt * dx;
        
        % Accumulate cost (integrated squared error of cart position and pendulum angle):
        y = [x(1,k); x(3,k)];  % desired is zero
        J = J + sum(y.^2)*dt;
    end
    
    % Penalize if final error is high:
    if norm([x(1,end); x(3,end)]) > 0.05
        J = J + 1e4;
    end
    if ~isfinite(J)
        J = 1e6;
    end
end

%% Final Simulation with Optimized Parameters
function simulateController(p_opt)
    % p_opt = [w0, lambda_x, lambda_theta, alpha_weight, K0, eta, epsilon, gamma]
    
    % Fixed system parameters:
    m_w = 0.432;  m_p = 5.0;
    I_w = 0.5 * (0.432) * (0.0726)^2;
    I_p = 5 * (0.4)^2;
    l = 0.4;  r = 0.0726;  g = 9.81;
    a = 2*m_w + m_p + 2*I_w/(r^2);
    b = m_p * l;
    c = m_p * l^2 + I_p;
    d = m_p * g * l;
    Delta = a*c - b^2;
    b0 = -b / Delta;
    
    % Extract optimized parameters:
    w0 = p_opt(1);
    lambda_x = p_opt(2);
    lambda_theta = p_opt(3);
    alpha_weight = p_opt(4);
    K_val = p_opt(5);
    eta = p_opt(6);
    epsilon = p_opt(7);
    gamma = p_opt(8);
    
    % Compute ESO gains:
    L1 = 3*w0;
    L2 = 3*w0^2;
    L3 = w0^3;
    
    dt = 0.001;
    T_final = 10;
    t = 0:dt:T_final;
    N = length(t);
    x = zeros(4, N);
    x(:,1) = [0; 0; 0.1; 0];
    u_total = zeros(1, N);
    K_hist = zeros(1, N);
    K_hist(1) = K_val;
    
    % ESO initialization:
    z_x = x(1,1);  z_xdot = x(2,1);  z_dist_x = 0;
    z_theta = x(3,1);  z_thetadot = x(4,1);  z_dist_theta = 0;
    
    for k = 1:N-1
        x_meas = x(:,k);
        cart_pos = x_meas(1);
        cart_vel = x_meas(2);
        theta = x_meas(3);
        theta_dot = x_meas(4);
        
        % ESO Updates:
        e_x_obs = cart_pos - z_x;
        z_x = z_x + dt*(z_xdot + L1*e_x_obs);
        z_xdot = z_xdot + dt*(z_dist_x + L2*e_x_obs);
        z_dist_x = z_dist_x + dt*(L3*e_x_obs);
        
        e_theta_obs = theta - z_theta;
        z_theta = z_theta + dt*(z_thetadot + L1*e_theta_obs);
        z_thetadot = z_thetadot + dt*(z_dist_theta + L2*e_theta_obs);
        z_dist_theta = z_dist_theta + dt*(L3*e_theta_obs);
        
        % Sliding surfaces:
        e_x = 0 - z_x;
        e_xdot = 0 - z_xdot;
        e_theta = 0 - z_theta;
        e_thetadot = 0 - z_thetadot;
        s_x = e_x + lambda_x * e_xdot;
        s_theta = e_theta + lambda_theta * e_thetadot;
        s = alpha_weight * s_x + (1 - alpha_weight) * s_theta;
        
        % Control Law:
        u_nom = - K_val * s;
        u_robust = - eta * tanh(s/epsilon);
        u_ASMC = u_nom + u_robust;
        u = (u_ASMC - (z_dist_x + z_dist_theta)) / b0;
        u = max(min(u, 10), -10);
        u_total(k) = u;
        
        % Adaptive Gain Update:
        K_val = K_val + gamma * s^2 * dt;
        K_hist(k+1) = K_val;
        
        % Plant Dynamics:
        denom = a*c - b^2 * cos(theta)^2;
        ddx = ( c*( u + b*sin(theta)*theta_dot^2 ) - b*cos(theta)*( m_p*g*l*sin(theta) ) ) / denom;
        ddtheta = ( a*(m_p*g*l*sin(theta)) - b*cos(theta)*( u + b*sin(theta)*theta_dot^2 ) ) / denom;
        dx = [ cart_vel; ddx; theta_dot; ddtheta ];
        x(:,k+1) = x(:,k) + dt * dx;
    end
    
    %% Plot Final Results
    figure;
    subplot(3,1,1);
    plot(t, x(1,:), 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Cart Position (m)');
    title('Optimized ADRC/SMC: Cart Position');
    grid on;
    
    subplot(3,1,2);
    plot(t, x(3,:), 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Pendulum Angle (rad)');
    title('Optimized ADRC/SMC: Pendulum Angle');
    grid on;
    
    subplot(3,1,3);
    plot(t(1:end-1), u_total(1:end-1), 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Control Input (N)');
    title('Optimized ADRC/SMC: Control Effort');
    grid on;
    
    figure;
    plot(t, K_hist, 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Adaptive Gain K');
    title('Evolution of Adaptive Gain with Optimized Parameters');
    grid on;
end
