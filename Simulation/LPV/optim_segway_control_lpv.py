import numpy as np
import control as ctrl
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import mujoco as mj
from mujoco.glfw import glfw
import time

# ----------------------------
# System and Model Parameters
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)   # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)           # moment of inertia of the pendulum [kg*m^2]
l  = 0.4           # length of the pendulum [m]
r  = 0.0726        # wheel radius [m]
g  = 9.81          # gravitational acceleration [m/s^2]

# Derived parameters
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Nominal state-space model (for p=0)
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],
              [0]])

# ----------------------------
# Simulation Parameters for GA Cost Function
# ----------------------------
T_sim = 5.0   # simulation time [s] for GA evaluation
dt = 0.01     # simulation time step [s]
numSched = 5  # number of scheduling points for LPV gain design

# ----------------------------
# GA Cost Function
# ----------------------------
def lpv_cost_function(params):
    """
    params = [Q1, Q2, Q3, Q4, R, p_min, p_max]
    The cost is computed as the integrated squared error of the outputs
    (cart position and pendulum angle) for the closed-loop LPV system.
    """
    Q_vals = params[0:4]
    R_val = params[4]
    p_min = params[5]
    p_max = params[6]
    
    # Ensure that p_min < p_max
    if p_min >= p_max:
        return 1e6,
    
    Q_mat = np.diag(Q_vals)
    R_mat = np.array([[R_val]])
    
    # Create scheduling range for LPV design
    p_range = np.linspace(p_min, p_max, numSched)
    K_schedule = np.zeros((numSched, 4))  # each K is 1x4
    
    for i, p_val in enumerate(p_range):
        # Effective gravity modification (small angle approximation)
        d_eff = mp * g * l * np.cos(p_val)
        A_lpv = A.copy()
        A_lpv[3, 2] = a * d_eff / Delta
        try:
            K, _, _ = ctrl.lqr(A_lpv, B, Q_mat, R_mat)
        except Exception as e:
            return 1e6,
        K_schedule[i, :] = np.asarray(K).flatten()
    
    # Create interpolation function for K gains based on current pendulum angle
    K_interp = interp1d(p_range, K_schedule, axis=0, fill_value="extrapolate")
    
    # Simulate closed-loop system using Euler integration
    N = int(T_sim/dt) + 1
    x = np.zeros((4, N))
    x[:, 0] = [0, 0, 0.1, 0]  # initial state: small pendulum deviation
    J = 0.0  # cost accumulator
    max_state_norm = 1e6  # safety threshold
    
    for k in range(N-1):
        p_current = x[2, k]  # scheduling parameter = pendulum angle
        K_current = K_interp(p_current)
        u = -np.dot(K_current, x[:, k])
        x_dot = np.dot(A, x[:, k]) + B.flatten() * u
        x[:, k+1] = x[:, k] + dt * x_dot
        
        # Safety check: abort if state becomes too large or NaN
        if np.linalg.norm(x[:, k+1]) > max_state_norm or np.any(np.isnan(x[:, k+1])):
            return 1e6,
        
        y = np.dot(C, x[:, k])
        e = y  # error (desired output is zero)
        J += np.sum(e**2) * dt
    return J,

# ----------------------------
# GA Setup Using DEAP
# ----------------------------
# Decision vector: [Q1, Q2, Q3, Q4, R, p_min, p_max]
# Bounds:
#   Q1: [1, 10000]
#   Q2: [1, 1000]
#   Q3: [1, 10000]
#   Q4: [1, 1000]
#   R: [0.001, 1]
#   p_min: [-0.2, -0.001]
#   p_max: [0.001, 0.2]
bounds = [(1, 10000), (1, 1000), (1, 10000), (1, 1000), (0.001, 1), (-0.2, -0.001), (0.001, 0.2)]
NDIM = 7

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

for i, (low, high) in enumerate(bounds):
    toolbox.register("attr_{}".format(i), random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2,
                  toolbox.attr_3, toolbox.attr_4, toolbox.attr_5, toolbox.attr_6), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lpv_cost_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# Use a higher eta to reduce chance of complex numbers
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------------
# GA Optimization Function
# ----------------------------
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
    print("p_min = {:.4f} rad, p_max = {:.4f} rad".format(best_params[5], best_params[6]))
    return best_params

# ----------------------------
# Build LPV Controller from Optimized Parameters
# ----------------------------
def build_lpv_controller(best_params):
    Q_opt = np.diag(best_params[0:4])
    R_opt = best_params[4]
    p_min_opt = best_params[5]
    p_max_opt = best_params[6]
    p_range_opt = np.linspace(p_min_opt, p_max_opt, numSched)
    K_schedule = np.zeros((numSched, 4))
    for i, p_val in enumerate(p_range_opt):
        d_eff = mp * g * l * np.cos(p_val)
        A_lpv = A.copy()
        A_lpv[3, 2] = a * d_eff / Delta
        K, _, _ = ctrl.lqr(A_lpv, B, Q_opt, np.array([[R_opt]]))
        K_schedule[i, :] = np.asarray(K).flatten()
    # Create an interpolation function that returns the gain vector for any pendulum angle
    K_interp = interp1d(p_range_opt, K_schedule, axis=0, fill_value="extrapolate")
    return K_interp

# ----------------------------
# MuJoCo Simulation Function
# ----------------------------
def run_mujoco_simulation(K_interp):
    # Load MuJoCo model (ensure segway.xml is in your working directory)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions:
    # data.qpos[0]: cart position; data.qpos[1]: pendulum angle.
    data.qpos[0] = 0.0   # cart position [m]
    data.qpos[1] = 0.1   # pendulum angle [rad]
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0

    # Identify the pendulum body index (if you want to apply disturbance, etc.)
    pendulum_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pendulum")
    
    # Simulation parameters
    duration = 10.0      # simulation duration in seconds
    dt_sim = model.opt.timestep  # MuJoCo simulation timestep
    time_sim = 0.0

    # Data logging lists
    time_data = []
    state_data = []
    control_data = []
    
    # Initialize MuJoCo Visualization (GLFW)
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    window = glfw.create_window(1200, 900, "MuJoCo LPV Simulation", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create window")
        return
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # Simulation Loop
    while time_sim < duration:
        # Retrieve current state: x = [cart pos, cart vel, pendulum angle, pendulum angular vel]
        x = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        # Use the pendulum angle as the scheduling parameter
        p_current = x[2]
        K_current = K_interp(p_current)
        u = -np.dot(K_current, x)  # control input
        data.ctrl[0] = u.item()
        
        # Step simulation
        mj.mj_step(model, data)
        
        # Log data
        time_data.append(time_sim)
        state_data.append(x.copy())
        control_data.append(u.item())
        
        # Render scene
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        time_sim += dt_sim
        if glfw.window_should_close(window):
            break

    glfw.terminate()
    
    # Optional: Plot simulation results
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_data, state_data[:, 0], 'b', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart Position")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_data, state_data[:, 2], 'r', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_data, control_data, 'g', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (N)")
    plt.title("Control Effort")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Function: Run GA, Build Controller, and Simulate in MuJoCo
# ----------------------------
def main():
    best_params = optimize_parameters()
    K_interp = build_lpv_controller(best_params)
    run_mujoco_simulation(K_interp)

if __name__ == "__main__":
    main()
