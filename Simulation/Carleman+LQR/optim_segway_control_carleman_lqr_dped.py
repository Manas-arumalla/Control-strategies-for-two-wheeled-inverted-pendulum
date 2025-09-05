import numpy as np
import control as ctrl      # pip install control (requires slycot for LQR)
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import random
import mujoco as mj
from mujoco.glfw import glfw
import time

# =============================================================
# 1. Physical and Derived Parameters (constants)
# =============================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
r = 0.0726         # wheel radius [m]
l = 0.4            # pendulum length [m]
g = 9.81           # gravitational acceleration [m/s^2]
Iw = 0.5 * mw * (r**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (l**2)       # moment of inertia of the pendulum [kg*m^2]

# Derived parameters for the non–augmented model:
a = 2 * mw + mp + 2 * Iw/(r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l  # shorthand for mp*g*l
denom = a * c - b**2

# =============================================================
# 2. Cost Function for GA: Carleman LQR Parameter Tuning
# =============================================================
def settling_cost_carleman(params):
    """
    Candidate decision vector (length 8):
      [Q1, Q2, Q3, Q4, Q5, R, alpha, delta]
      
    where the augmented LQR is designed for the augmented state:
      [x, x_dot, theta, theta_dot, z5]
      
    Q is defined as diag([Q1, Q2, Q3, Q4, Q5]) and R is scalar.
    The auxiliary (Carleman) dynamics are governed by:
      z5_dot = theta_dot + alpha*(theta - z5) + delta*u.
      
    The cost is computed as the integrated squared error of the physical states
    (first 4 entries) over a simulation time T_sim.
    If the simulation diverges, a high cost is returned.
    """
    Q1, Q2, Q3, Q4, Q5, R_val, alpha_val, delta_val = params
    
    # Construct the augmented matrices using candidate parameters.
    Q_aug = np.diag([Q1, Q2, Q3, Q4, Q5])
    R_mat = np.array([[R_val]])
    
    # Augmented dynamics (continuous-time)
    A_aug = np.array([
        [0,   1,               0,               0,       0],
        [0,   0,      -b*d/denom,               0,       0],
        [0,   0,               0,               1,       0],
        [0,   0,       a*d/denom,               0,       0],
        [0,   0,         alpha_val,            1, -alpha_val]
    ])
    B_aug = np.array([
        [0],
        [c/denom],
        [0],
        [-b/denom],
        [delta_val]
    ])
    
    # Try to compute the LQR gain.
    try:
        K, _, _ = ctrl.lqr(A_aug, B_aug, Q_aug, R_mat)
        K = np.asarray(K)
    except Exception as e:
        # If LQR fails, return a high cost.
        return 1e6,
    
    # ---------------------------------------------------------
    # Simulation settings for cost evaluation:
    T_sim = 5.0      # total simulation time [s]
    dt_sim = 0.01    # simulation timestep
    N = int(T_sim/dt_sim) + 1
    # Initial condition: small disturbance in pendulum angle.
    # Augmented state: [x, x_dot, theta, theta_dot, z5]
    z = np.array([0.0, 0.0, 0.2, 0.0, 0.2])
    
    cost = 0.0
    for k in range(N):
        # Compute control input: u = -K*z
        u = -np.dot(K, z).item()
        # Compute state derivative:
        z_dot = A_aug @ z + (B_aug.flatten() * u)
        # Euler integration:
        z = z + dt_sim * z_dot
        
        # Physical states are the first four entries.
        x_physical = z[:4]
        cost += (np.linalg.norm(x_physical)**2) * dt_sim
        
        # If the state diverges, penalize heavily.
        if np.linalg.norm(z) > 1e6 or np.any(np.isnan(z)):
            return 1e6,
    
    return cost,

# =============================================================
# 3. GA Setup Using DEAP for Carleman LQR Tuning
# =============================================================
# Decision vector: 8 parameters.
# Parameter bounds:
bounds = [
    (50, 150),      # Q1
    (0.5, 2),       # Q2
    (50, 150),      # Q3
    (0.5, 2),       # Q4
    (5, 15),        # Q5
    (0.1, 10),      # R
    (0.1, 2.0),     # alpha
    (0.01, 0.1)     # delta
]
NDIM = len(bounds)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

# Register attribute generators.
for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

# Register individual and population generators.
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                  toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", settling_cost_carleman)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def safe_mutate(individual, eta, low, up, indpb):
    try:
        mutated_ind, = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    except TypeError:
        individual[:] = [float(np.real(x)) for x in individual]
        mutated_ind, = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    for i, val in enumerate(mutated_ind):
        mutated_ind[i] = float(np.real(val))
    return mutated_ind,

toolbox.register("mutate", safe_mutate,
                 eta=20, low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
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
    print("\nOptimized Carleman LQR Parameters:")
    print("LQR Q = diag([{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]), R = {:.4f}".format(
          best_params[0], best_params[1], best_params[2], best_params[3], best_params[4], best_params[5]))
    print("Auxiliary dynamics tuning: alpha = {:.4f}, delta = {:.4f}".format(
          best_params[6], best_params[7]))
    return best_params

# =============================================================
# 4. MuJoCo Simulation with Optimized Carleman LQR Controller
# =============================================================
def run_mujoco_simulation(opt_params):
    # Unpack optimized parameters.
    Q1, Q2, Q3, Q4, Q5, R_val, alpha_val, delta_val = opt_params
    Q_aug = np.diag([Q1, Q2, Q3, Q4, Q5])
    R_mat = np.array([[R_val]])
    
    # Rebuild the augmented state-space matrices.
    A_aug = np.array([
        [0,   1,               0,               0,       0],
        [0,   0,      -b*d/denom,               0,       0],
        [0,   0,               0,               1,       0],
        [0,   0,       a*d/denom,               0,       0],
        [0,   0,         alpha_val,            1, -alpha_val]
    ])
    B_aug = np.array([
        [0],
        [c/denom],
        [0],
        [-b/denom],
        [delta_val]
    ])
    
    try:
        K, _, _ = ctrl.lqr(A_aug, B_aug, Q_aug, R_mat)
        K = np.asarray(K)
    except Exception as e:
        print("LQR computation failed with optimized parameters.")
        return
    
    print("\nOptimized LQR Gain K:")
    print(K)
    
    # Load MuJoCo model (ensure segway.xml is in the working directory).
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions:
    data.qpos[0] = 0.0   # Base (slider) position (x)
    data.qpos[1] = 0.2   # Pendulum angle (theta) [rad]
    data.qvel[0] = 0.0   # Base velocity (x_dot)
    data.qvel[1] = 0.0   # Pendulum angular velocity (theta_dot)
    
    # Initialize auxiliary state z5 to the initial pendulum angle.
    z5 = data.qpos[1].item()
    
    duration = 10.0          # simulation duration [s]
    dt = model.opt.timestep  # simulation timestep
    t_sim = 0.0
    
    # Data recording lists.
    time_data = []
    state_data = []      # physical states: [x, x_dot, theta, theta_dot]
    z5_data = []         # auxiliary state history
    control_data = []    # control input history
    
    # Initialize MuJoCo visualization (GLFW).
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    window = glfw.create_window(1200, 900, "Carleman LQR with GA Tuning", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create GLFW window")
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    while t_sim < duration:
        # Extract physical states.
        x = data.qpos[0].item()
        theta = data.qpos[1].item()
        x_dot = data.qvel[0].item()
        theta_dot = data.qvel[1].item()
        
        # Form the augmented state vector: [x, x_dot, theta, theta_dot, z5].
        z = np.array([x, x_dot, theta, theta_dot, z5])
        # Compute the control input: u = -K*z.
        u = -np.dot(K, z).item()
        
        # Apply the control input.
        data.ctrl[0] = u
        
        # Update the auxiliary state z5 using Euler integration:
        # z5_dot = theta_dot + alpha*(theta - z5) + delta*u.
        z5_dot = theta_dot + alpha_val * (theta - z5) + delta_val * u
        z5 = z5 + dt * z5_dot
        
        # Step the simulation.
        mj.mj_step(model, data)
        
        # Record simulation data.
        time_data.append(t_sim)
        state_data.append([x, x_dot, theta, theta_dot])
        z5_data.append(z5)
        control_data.append(u)
        
        # Render the scene.
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt_vis, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        t_sim += dt
        if glfw.window_should_close(window):
            break
    
    glfw.terminate()
    
    # Plot the results.
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    z5_data = np.array(z5_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_data, state_data[:, 0], 'b', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.title("Base (Slider) Position")
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time_data, state_data[:, 2], 'r', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(time_data, z5_data, 'm', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("z5")
    plt.title("Auxiliary State z5")
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time_data, control_data, 'g', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (N)")
    plt.title("LQR Control Input")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================
# 5. Main: Run GA Optimization then MuJoCo Simulation
# =============================================================
def main():
    print("Starting GA optimization for Carleman LQR tuning...")
    best_params = optimize_parameters()
    print("Running MuJoCo simulation with optimized parameters...")
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
