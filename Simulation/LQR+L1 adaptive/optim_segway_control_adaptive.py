import numpy as np
import control as ctrl
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import mujoco as mj
from mujoco.glfw import glfw
import time

# ============================
# 1. System & Model Parameters
# ============================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)   # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)           # moment of inertia of the pendulum [kg*m^2]
l = 0.4            # length of the pendulum [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Nominal state-space model (continuous-time) for the plant
# State: x = [ cart position; cart velocity; pendulum angle; pendulum angular velocity ]
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])
# (For the simulation cost function we use the full state vector for cost accumulation.)

# ====================================
# 2. GA Optimization: Define the Cost Function
# ====================================
def l1_cost_function(params):
    """
    Candidate parameters (decision vector):
      [Q1, Q2, Q3, Q4, R, Gamma, tau, sigma_max]
    where:
      - Q1,Q2,Q3,Q4: Diagonal entries of Q used in LQR design.
      - R: Scalar weight.
      - Gamma: Adaptation gain.
      - tau: Low-pass filter time constant.
      - sigma_max: Projection bound.
    
    The cost is computed by simulating the closed-loop dynamics 
    of the plant with L1 adaptive control (using Euler integration) 
    and accumulating the integrated squared error of the state.
    """
    # Unpack parameters
    Q_vals = params[0:4]
    R_val = params[4]
    Gamma = params[5]
    tau = params[6]
    sigma_max = params[7]
    
    # Build LQR design matrices
    Q_mat = np.diag(Q_vals)
    R_mat = np.array([[R_val]])
    try:
        K, _, _ = ctrl.lqr(A, B, Q_mat, R_mat)
    except Exception as e:
        return 1e6,  # If LQR fails, return a high cost.
    K = np.asarray(K)
    
    # Closed-loop nominal dynamics for computing the Lyapunov matrix P:
    A_cl = A - B @ K
    try:
        P = solve_continuous_lyapunov(A_cl.T, -np.eye(4))
    except Exception as e:
        return 1e6,
    
    # Simulation parameters
    T_sim = 10.0    # total simulation time [s]
    dt_sim = 0.001  # simulation timestep [s]
    N = int(T_sim / dt_sim) + 1
    
    # Initialize states for the plant and predictor, and the adaptive variables.
    x = np.zeros((4, N))
    x_hat = np.zeros((4, N))
    sigma_hat = 0.0
    z_f = 0.0
    x[:, 0] = [0, 0, 0.1, 0]  # initial plant state (small pendulum deviation)
    
    cost = 0.0
    max_state_norm = 1e6  # safety threshold
    
    # Simulation loop (Euler integration)
    for k in range(N - 1):
        # Control law: u = -K*x - z_f
        u = -np.dot(K, x[:, k]) - z_f
        
        # Plant dynamics: x_dot = A*x + B*u
        x_dot = np.dot(A, x[:, k]) + B.flatten() * u
        x[:, k+1] = x[:, k] + dt_sim * x_dot
        
        # Predictor dynamics: x_hat_dot = A*x_hat + B*(u + sigma_hat)
        x_hat_dot = np.dot(A, x_hat[:, k]) + B.flatten() * (u + sigma_hat)
        x_hat[:, k+1] = x_hat[:, k] + dt_sim * x_hat_dot
        
        # Adaptation law:
        sigma_dot_temp = Gamma * (B.T @ P @ (x[:, k] - x_hat[:, k])).item()
        if (sigma_hat >= sigma_max and sigma_dot_temp > 0) or (sigma_hat <= -sigma_max and sigma_dot_temp < 0):
            sigma_dot = 0.0
        else:
            sigma_dot = sigma_dot_temp
        sigma_hat = sigma_hat + dt_sim * sigma_dot
        
        # Low-pass filter dynamics: z_f_dot = (-1/tau)*z_f + (1/tau)*sigma_hat
        z_f_dot = (-1.0/tau)*z_f + (1.0/tau)*sigma_hat
        z_f = z_f + dt_sim * z_f_dot
        
        # Accumulate cost (here, we penalize state error)
        cost += np.sum(x[:, k]**2) * dt_sim
        
        # Safety check: abort if state becomes unreasonably large or NaN.
        if np.linalg.norm(x[:, k+1]) > max_state_norm or np.any(np.isnan(x[:, k+1])):
            return 1e6,
    
    return cost,

# ======================================================
# 3. GA Setup Using DEAP to Optimize All Tunable Parameters
# ======================================================
# Decision vector: [Q1, Q2, Q3, Q4, R, Gamma, tau, sigma_max]
# Suggested bounds (tune these as needed):
#   Q1: [1, 10000]
#   Q2: [1, 1000]
#   Q3: [1, 10000]
#   Q4: [1, 1000]
#   R: [0.001, 1]
#   Gamma: [1, 100]
#   tau: [0.01, 1]
#   sigma_max: [1, 20]
bounds = [(1, 10000), (1, 1000), (1, 10000), (1, 1000), (0.001, 1), (1, 100), (0.01, 1), (1, 20)]
NDIM = 8

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2,
                  toolbox.attr_3, toolbox.attr_4, toolbox.attr_5,
                  toolbox.attr_6, toolbox.attr_7), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", l1_cost_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# Use polynomial bounded mutation with a high eta to keep mutations smooth.
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_parameters():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_params = hof[0]
    print("\nOptimized Parameters:")
    print("Q = diag([{:.2f}, {:.2f}, {:.2f}, {:.2f}])".format(best_params[0], best_params[1],
                                                              best_params[2], best_params[3]))
    print("R = {:.4f}".format(best_params[4]))
    print("Gamma = {:.2f}".format(best_params[5]))
    print("tau = {:.4f}".format(best_params[6]))
    print("sigma_max = {:.2f}".format(best_params[7]))
    return best_params

# ======================================================
# 4. MuJoCo Simulation Using Optimized Parameters
# ======================================================
def run_mujoco_simulation(best_params):
    # Unpack the optimized parameters for the LQR and L1 controller.
    Q_opt = np.diag(best_params[0:4])
    R_opt = np.array([[best_params[4]]])
    Gamma_opt = best_params[5]
    tau_opt = best_params[6]
    sigma_max_opt = best_params[7]
    
    # Compute the nominal LQR gain.
    K, _, _ = ctrl.lqr(A, B, Q_opt, R_opt)
    K = np.asarray(K)
    A_cl = A - B @ K
    P = solve_continuous_lyapunov(A_cl.T, -np.eye(4))
    
    # Initialize adaptive controller states.
    x_hat = np.zeros((4,))
    sigma_hat = 0.0
    z_f = 0.0
    
    # Load the MuJoCo model.
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions:
    data.qpos[0] = 0.0   # cart position [m]
    data.qpos[1] = 0.1   # pendulum angle [rad]
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    # Identify the pendulum body index (for applying disturbance if desired)
    pendulum_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pendulum")
    
    # Simulation parameters
    duration = 10.0      # simulation duration [s]
    dt_sim = model.opt.timestep  # simulation timestep (e.g., 0.001 s)
    t_sim = 0.0
    
    # Data logging
    time_data = []
    state_data = []
    control_data = []
    sigma_data = []
    zf_data = []
    
    # Initialize MuJoCo visualization (GLFW)
    if not glfw.init():
        print("GLFW initialization failed")
        return
    window = glfw.create_window(1200, 900, "Optimized L1 Adaptive Controller", None, None)
    if not window:
        glfw.terminate()
        print("Window creation failed")
        return
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # Simulation loop
    while t_sim < duration:
        # Get current plant state: [cart pos, cart vel, pendulum angle, pendulum angular vel]
        x = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        # Control law with L1 adaptive term: u = -K*x - z_f
        u = -np.dot(K, x) - z_f
        
        # (Optional) Apply an external disturbance torque on the pendulum
        d_ext = 0.5 * np.sin(0.5 * t_sim)
        data.xfrc_applied[pendulum_body_id, :] = np.array([0, 0, 0, 0, d_ext, 0])
        
        # Apply control input to the actuator (assume first actuator controls the cart)
        data.ctrl[0] = u.item()
        
        # Step the simulation
        mj.mj_step(model, data)
        
        # Update the L1 adaptive controller dynamics using Euler integration:
        # Predictor dynamics: x_hat_dot = A*x_hat + B*(u + sigma_hat)
        x_hat_dot = np.dot(A, x_hat) + B.flatten()*(u + sigma_hat)
        x_hat = x_hat + dt_sim * x_hat_dot
        
        # Adaptation law:
        sigma_dot_temp = Gamma_opt * (B.T @ P @ (x - x_hat)).item()
        if (sigma_hat >= sigma_max_opt and sigma_dot_temp > 0) or (sigma_hat <= -sigma_max_opt and sigma_dot_temp < 0):
            sigma_dot = 0.0
        else:
            sigma_dot = sigma_dot_temp
        sigma_hat = sigma_hat + dt_sim * sigma_dot
        
        # Low-pass filter dynamics:
        z_f_dot = (-1.0/tau_opt)*z_f + (1.0/tau_opt)*sigma_hat
        z_f = z_f + dt_sim * z_f_dot
        
        # Logging data
        time_data.append(t_sim)
        state_data.append(x.copy())
        control_data.append(u.item())
        sigma_data.append(sigma_hat)
        zf_data.append(z_f)
        
        # Render the scene
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        t_sim += dt_sim
        if glfw.window_should_close(window):
            break

    glfw.terminate()
    
    # Plot simulation results
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    sigma_data = np.array(sigma_data)
    zf_data = np.array(zf_data)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(time_data, state_data[:, 0], label="Cart Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Cart Position")
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time_data, state_data[:, 2], label="Pendulum Angle", color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(time_data, control_data, label="Control Input", color="g")
    plt.xlabel("Time (s)")
    plt.ylabel("Control (N)")
    plt.title("Control Effort")
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time_data, sigma_data, 'm', label="Adaptive Estimate")
    plt.plot(time_data, zf_data, 'c--', label="Filtered Term")
    plt.xlabel("Time (s)")
    plt.ylabel("Adaptive Terms")
    plt.title("Adaptive Estimate (σ̂) and Filtered Term (z_f)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ======================================================
# 5. Main: Optimize and Simulate
# ======================================================
def main():
    best_params = optimize_parameters()
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
