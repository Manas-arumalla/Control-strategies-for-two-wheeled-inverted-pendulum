import numpy as np
import control as ctrl      # pip install control (requires slycot for LQR)
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw
import time

# =============================================================
# 1. Physical and Derived Parameters
# =============================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # Inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)          # Inertia of the pendulum [kg*m^2]
l   = 0.4          # Distance from axle to pendulum COM [m]
r   = 0.0726       # Wheel radius [m]
g   = 9.81         # Gravity [m/s^2]

# Coefficients from the dynamics derivation
a = 2 * mw + mp + 2 * Iw/(r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a*c - b**2

# =============================================================
# 2. Linearized Model for NMPC (state vector: [x, x_dot, theta, theta_dot])
# =============================================================
A_lin = np.array([[0,     1,                     0,         0],
                  [0,     0,           -b*d/Delta,         0],
                  [0,     0,                     0,         1],
                  [0,     0,           a*d/Delta,         0]])
B_lin = np.array([[0],
                  [c/Delta],
                  [0],
                  [-b/Delta]])

# =============================================================
# 3. Controller Tuning Parameters
# =============================================================
# NMPC parameters
N_horizon = 10           # Prediction horizon (steps)
Q_nmpc = np.diag([10, 1, 100, 1])
R_nmpc = 0.01            # Control input weighting
u_sat = 10.0             # Saturation limit on control input

# DSC (Dynamic Surface Control) parameters
lambda_dsc = 3.0         # DSC filter coefficient
alpha_dsc  = 20.0        # DSC gain
z_filter = 0.0           # Filter state for DSC (initialize to zero)

# Adaptive NN parameters (using a scalar weight)
nn_w = 0.0               # Initial NN weight
gamma_nn = 0.05          # NN weight update rate
nn_w_max = 5.0
nn_w_min = -5.0

# =============================================================
# 4. NMPC Cost Function
# =============================================================
def nmpc_cost(u_seq, x0, A, B, Q, R, N_horizon):
    """
    Computes the quadratic cost over the prediction horizon.
    u_seq: control sequence candidate (vector of length N_horizon)
    x0: current state (4-vector)
    A, B: linearized model matrices
    Q: state weighting matrix
    R: scalar control weighting
    """
    u_seq = np.reshape(u_seq, (N_horizon,))
    x_pred = x0.copy()
    cost = 0.0
    for k in range(N_horizon):
        u = u_seq[k]
        x_pred = A.dot(x_pred) + B.flatten() * u
        cost += x_pred.T.dot(Q).dot(x_pred) + R*(u**2)
    if not np.isfinite(cost):
        cost = 1e6
    return cost

# =============================================================
# 5. Main Simulation Loop: NMPC+DSC+NN Hybrid Controller
# =============================================================
def run_simulation():
    T_final = 5.0         # Final simulation time [s]
    dt = 0.001            # Time step [s]
    t_vec = np.arange(0, T_final+dt, dt)
    N = len(t_vec)
    
    # Initialize state vector (x, x_dot, theta, theta_dot)
    x = np.zeros((4, N))
    x[:, 0] = [0.0, 0.0, 0.1, 0.0]   # small initial pendulum disturbance
    
    # Preallocate control input storage
    u_total_arr = np.zeros(N)
    
    global z_filter, nn_w  # use the global DSC filter state and NN weight

    # Simulation loop
    for k in range(N-1):
        x_current = x[:, k]
        
        # ----- Outer-Loop NMPC -----
        u0_seq = np.zeros(N_horizon)
        bounds_nmpc = [(-u_sat, u_sat)] * N_horizon
        sol = minimize(nmpc_cost, u0_seq, args=(x_current, A_lin, B_lin, Q_nmpc, R_nmpc, N_horizon),
                       bounds=bounds_nmpc, method='SLSQP', options={'disp': False})
        u_nmpc = sol.x[0]  # Use only the first control action
        
        # ----- Inner-Loop DSC -----
        e = x_current[2]  # Pendulum angle error (desired = 0)
        e_dot = x_current[3]  # Angular velocity
        # Update DSC filter state (first-order filter)
        z_filter = (1 - dt*lambda_dsc)*z_filter + dt * e_dot
        u_dsc = -alpha_dsc * (e + z_filter)
        
        # ----- Adaptive NN Augmentation -----
        u_nn = nn_w * e
        # Update NN weight (gradient descent on e^2)
        nn_w = nn_w + gamma_nn * (e**2)
        nn_w = np.clip(nn_w, nn_w_min, nn_w_max)
        
        # ----- Combine Control Inputs and Saturate -----
        u_combined = u_nmpc + u_dsc + u_nn
        u_total = np.clip(u_combined, -u_sat, u_sat)
        u_total_arr[k] = u_total
        
        # ----- Full Nonlinear Dynamics Simulation -----
        # Extract current state variables
        pos = x_current[0]         # cart position
        vel = x_current[1]         # cart velocity
        theta = x_current[2]       # pendulum angle
        theta_dot = x_current[3]   # pendulum angular velocity
        
        # Nonlinear dynamics (from the derivation):
        denom = a*c - b**2 * np.cos(theta)**2
        ddx = ( c*(u_total + b*np.sin(theta)*theta_dot**2) - b*np.cos(theta)*(mp*g*l*np.sin(theta)) ) / denom
        ddtheta = ( a*(mp*g*l*np.sin(theta)) - b*np.cos(theta)*(u_total + b*np.sin(theta)*theta_dot**2) ) / denom
        
        x_dot = np.array([ vel,
                           ddx,
                           theta_dot,
                           ddtheta ])
        x[:, k+1] = x_current + dt * x_dot

    return t_vec, x, u_total_arr

# =============================================================
# 6. MuJoCo Simulation with Hybrid NMPC+DSC+NN Controller
# =============================================================
def run_mujoco_simulation():
    # Run the hybrid controller simulation first to get a reference plot.
    t_sim, state_sim, control_sim = run_simulation()
    
    # Plot the simulation results (nonlinear simulation)
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3,1,1)
    plt.plot(t_sim, state_sim[0, :], 'b', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart (Slider) Position")
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(t_sim, state_sim[2, :], 'r', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.plot(t_sim, control_sim, 'k', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (N)")
    plt.title("Combined Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ----- Now, run the same controller in a MuJoCo simulation -----
    # Reset global filter state and NN weight (if needed)
    global z_filter, nn_w
    z_filter = 0.0
    nn_w = 0.0

    # Load MuJoCo model (ensure "segway.xml" is in your working directory)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions (adjust as needed)
    data.qpos[0] = 0.0    # cart position [m]
    data.qpos[1] = 0.1    # pendulum angle [rad]
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0

    duration = 10.0
    dt = model.opt.timestep
    t_sim_mj = 0.0

    time_data = []
    state_data = []   # [cart pos, cart vel, pend angle, pend ang vel]
    control_data = []
    
    # Initialize visualization using GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "Hybrid NMPC+DSC+NN Controlled Segway", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # Main simulation loop in MuJoCo
    while t_sim_mj < duration:
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        # Outer-loop NMPC
        u0_seq = np.zeros(N_horizon)
        bounds_nmpc = [(-u_sat, u_sat)] * N_horizon
        sol = minimize(nmpc_cost, u0_seq, args=(state, A_lin, B_lin, Q_nmpc, R_nmpc, N_horizon),
                       bounds=bounds_nmpc, method='SLSQP', options={'disp': False})
        u_nmpc = sol.x[0]
        
        # Inner-loop DSC
        e = state[2]
        e_dot = state[3]
        z_filter = (1 - dt*lambda_dsc)*z_filter + dt*e_dot
        u_dsc = -alpha_dsc * (e + z_filter)
        
        # Adaptive NN
        u_nn = nn_w * e
        nn_w = nn_w + gamma_nn * (e**2)
        nn_w = np.clip(nn_w, nn_w_min, nn_w_max)
        
        # Combined control and saturation
        u_raw = (1.0 * u_nmpc) + (1.0 * u_dsc) + (1.0 * u_nn)  # weights can be adjusted if desired
        filtered_u = (1 - 0.05)*0.0  # Initialize local filter variable
        # Here we use a simple one-step filter update:
        filtered_u = (1 - 0.05)*u_raw + 0.05*u_raw
        u_total = np.clip(u_raw, -u_sat, u_sat)
        
        control_data.append(u_total)
        
        # Apply control to MuJoCo model (assuming actuator index 0)
        data.ctrl[0] = u_total
        mj.mj_step(model, data)
        
        time_data.append(t_sim_mj)
        state_data.append(state.copy())
        
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt_vis, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        t_sim_mj += dt
        if glfw.window_should_close(window):
            break
    
    glfw.terminate()
    
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(3,1,1)
    plt.plot(time_data, state_data[:, 0], 'b', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart Position")
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(time_data, state_data[:, 2], 'r', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.plot(time_data, control_data, 'k', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (N)")
    plt.title("Combined Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================
# 7. Main Execution
# =============================================================
if __name__ == "__main__":
    print("Running NMPC+DSC+NN Hybrid Controller Simulation (offline)...")
    run_mujoco_simulation()
