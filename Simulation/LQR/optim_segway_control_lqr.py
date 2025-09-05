import numpy as np
import control as ctrl      # pip install control (requires slycot for LQR)
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import random
import mujoco as mj
from mujoco.glfw import glfw
import time

# ============================================================
# 1. Plant Definition (same as your provided parameters)
# ============================================================
# Physical parameters:
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # wheel inertia [kg*m^2]
Ip = 5.0 * (0.4**2)          # pendulum inertia [kg*m^2]
l = 0.4            # pendulum length [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters:
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Continuous-time state-space model (states: [x, x_dot, theta, theta_dot])
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

# ============================================================
# 2. GA Cost Function: Settling Time of Closed-Loop Response
# ============================================================
def settling_time_cost(params):
    """
    Decision vector:
      [Q1, Q2, Q3, Q4, R]
    where the LQR design uses Q = diag([Q1, Q2, Q3, Q4]) and scalar R.
    
    We simulate the closed-loop dynamics with u = -K x starting from an
    initial disturbance (e.g. x0 = [0, 0, 0.2, 0]) and compute the settling time.
    
    Settling time is defined as the earliest time at which the state norm stays
    below a threshold epsilon for the remainder of the simulation.
    
    If the state never settles or diverges, we return a high cost.
    """
    Q_vals = params[0:4]
    R_val = params[4]
    Q = np.diag(Q_vals)
    R = np.array([[R_val]])
    try:
        K, S, E = ctrl.lqr(A, B, Q, R)
        K = np.asarray(K)
    except Exception as e:
        return 1e6,  # if LQR synthesis fails, return a high cost
    
    # Closed-loop system: x_dot = (A - B*K)x
    A_cl = A - B @ K
    
    # Simulation parameters for cost evaluation:
    T_sim = 5.0        # simulation horizon [s]
    dt_sim = 0.01      # time step
    N = int(T_sim / dt_sim) + 1
    t = np.linspace(0, T_sim, N)
    
    # Initial condition (a disturbance in pendulum angle)
    x = np.zeros((4, N))
    x[:, 0] = [0.0, 0.0, 0.2, 0.0]
    
    # Define a threshold for "settled" state (e.g., state norm < epsilon)
    epsilon = 0.05
    settling_time = T_sim + 10  # default high cost if never settles
    
    # Euler integration simulation
    for k in range(N - 1):
        x[:, k+1] = x[:, k] + dt_sim * (A_cl @ x[:, k])
        # Safety: if state explodes, abort early
        if np.linalg.norm(x[:, k+1]) > 1e6 or np.any(np.isnan(x[:, k+1])):
            return 1e6,
    
    # Find the first time when the state remains below epsilon for the rest of simulation.
    for k in range(N):
        if np.linalg.norm(x[:, k]) < epsilon:
            # Check if for all future time steps, state norm remains below epsilon
            if np.all(np.linalg.norm(x[:, k:], axis=0) < epsilon):
                settling_time = t[k]
                break
    
    return settling_time,

# ============================================================
# 3. GA Setup Using DEAP for LQR Parameter Tuning
# ============================================================
# Decision vector: [Q1, Q2, Q3, Q4, R]
# Suggested bounds (tune as needed):
#   Q1 in [1, 1000]
#   Q2 in [1, 100]
#   Q3 in [1, 1000]
#   Q4 in [1, 100]
#   R in [0.001, 1]
bounds = [(1, 1000), (1, 100), (1, 1000), (1, 100), (0.001, 1)]
NDIM = 5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2,
                  toolbox.attr_3, toolbox.attr_4), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", settling_time_cost)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# We use the default polynomial bounded mutation; ensure candidates remain real.
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_lqr_parameters():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_params = hof[0]
    print("\nOptimized LQR Parameters:")
    print(f"Q = diag([{best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.2f}, {best_params[3]:.2f}])")
    print(f"R = {best_params[4]:.4f}")
    return best_params

# ============================================================
# 4. MuJoCo Simulation with the Optimized LQR Controller
# ============================================================
def run_mujoco_simulation(opt_params):
    # Unpack optimized parameters.
    Q_vals = opt_params[0:4]
    R_val = opt_params[4]
    Q = np.diag(Q_vals)
    R = np.array([[R_val]])
    
    # Compute LQR gain
    K, S, E = ctrl.lqr(A, B, Q, R)
    K = np.asarray(K)
    print("Optimized LQR Gain K:", K)
    
    # Target state is zero.
    x_ref = np.zeros((4,))
    
    # Load the MuJoCo model (ensure "segway.xml" is in your working directory)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions (apply a disturbance in pendulum angle)
    data.qpos[0] = 0.0    # cart position (m)
    data.qpos[1] = 0.2    # pendulum angle (rad)
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    duration = 10.0         # simulation time (s)
    dt = model.opt.timestep # simulation timestep (e.g., 0.001 s)
    t_sim = 0.0
    
    time_data = []
    state_data = []   # [cart pos, cart vel, pend angle, pend ang vel]
    control_data = []
    
    # Initialize MuJoCo visualization (GLFW)
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "Optimized LQR Controlled Segway", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # Simulation loop
    while t_sim < duration:
        # Extract current state:
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        # Compute control: u = -K*(state - x_ref)
        u = -np.dot(K, (state - x_ref))
        data.ctrl[0] = u.item()
        
        mj.mj_step(model, data)
        
        time_data.append(t_sim)
        state_data.append(state.copy())
        control_data.append(u.item())
        
        # Render the scene
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt_vis, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        t_sim += dt
        if glfw.window_should_close(window):
            break
    
    glfw.terminate()
    
    # Plot simulation results
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_data, state_data[:, 0], label="Cart Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Cart Position")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_data, state_data[:, 2], label="Pendulum Angle", color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_data, control_data, label="Control Input", color="g")
    plt.xlabel("Time (s)")
    plt.ylabel("Control (N)")
    plt.title("Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. Main: Run GA Optimization and then MuJoCo Simulation
# ============================================================
def main():
    best_params = optimize_lqr_parameters()
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
