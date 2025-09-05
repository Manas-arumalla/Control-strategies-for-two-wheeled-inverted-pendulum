import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import cont2discrete
from deap import base, creator, tools, algorithms
import random
import warnings

# Optional: Suppress SLSQP runtime warnings if desired.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize._slsqp_py")

# =============================================================
# 1. Plant Parameters and Discretization (from MATLAB derivation)
# =============================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)   # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)           # moment of inertia of the pendulum [kg*m^2]
l = 0.4            # pendulum length [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters (from linearization)
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * l**2 + Ip
d = mp * g * l
Delta = a * c - b**2

# Continuous-time state-space model (states: [cart pos; cart vel; pendulum angle; pendulum angular vel])
A_cont = np.array([[0, 1,           0,      0],
                   [0, 0,  -b*d/Delta,      0],
                   [0, 0,           0,      1],
                   [0, 0,   a*d/Delta,      0]])
B_cont = np.array([[0],
                   [c/Delta],
                   [0],
                   [-b/Delta]])
# Output matrix: we measure [cart position; pendulum angle]
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# Discretize using sampling time Ts (EPSAC update interval)
Ts = 0.1  # seconds
sysd = cont2discrete((A_cont, B_cont, C, np.zeros((2, 1))), Ts)
A_d = sysd[0]
B_d = sysd[1]

# Input constraints for control signal
u_min = -10
u_max = 10

# =============================================================
# 2. EPSAC Cost Function
# =============================================================
def epsac_cost(U, x0, A_est, B_est, Nc, Np, C, lam):
    """
    Computes the cost over the prediction horizon for a candidate control sequence U.
    U: vector of control moves (length Nc)
    x0: current state vector (4x1)
    The control sequence is extended (by holding the last value) to length Np.
    
    The cost is computed as the sum over the prediction horizon of a weighted squared error 
    (here we weight the pendulum angle error more heavily) plus a penalty on control effort.
    """
    # Ensure prediction horizon is at least as long as control horizon.
    if Np < Nc:
        return 1e6

    # Extend U to length Np by repeating the last value.
    U_extended = np.concatenate([U, np.full(Np - Nc, U[-1])])
    x_pred = x0.copy()
    cost = 0.0
    # Reference output is zero (i.e. zero cart displacement and zero pendulum angle deviation)
    ref = np.zeros((C.shape[0],))
    # Define a weight vector for output errors: less weight for cart position, high weight for pendulum angle.
    weight = np.array([1.0, 100.0])
    for i in range(Np):
        x_pred = A_est @ x_pred + B_est.flatten() * U_extended[i]
        y_pred = C @ x_pred
        e = ref - y_pred
        # Weighted squared error plus control effort penalty.
        cost += np.sum(weight * (e**2)) + lam * (U_extended[i]**2)
    return cost

# =============================================================
# 3. Simulation of EPSAC Control for Settling Time (GA Fitness Function)
# =============================================================
def simulate_epsac(params):
    """
    Simulate the EPSAC controller with the given parameters and return the settling time.
    
    Parameters (to be tuned by GA):
      - Np: Prediction horizon (rounded to integer and at least 1)
      - Nc: Control horizon (rounded to integer and at least 1)
      - lam: Weight on control effort
      - adaptation_gain: Gain for updating the estimated model matrices
    """
    # Unpack and enforce minimum values
    Np = max(1, int(round(params[0])))
    Nc = max(1, int(round(params[1])))
    lam_param = params[2]
    adaptation_gain = params[3]
    
    # If prediction horizon is less than control horizon, return high cost.
    if Np < Nc:
        return 1e6

    T_sim = 10.0           # Total simulation time [s]
    N_sim = int(T_sim / Ts)  # Number of EPSAC control updates
    threshold = 0.05       # Settling threshold (norm of state vector)

    # Initialize state trajectory: [cart pos, cart vel, pendulum angle, pendulum angular vel]
    x = np.zeros((4, N_sim + 1))
    # Use an initial condition with a small deviation in pendulum angle.
    x[:, 0] = np.array([0.0, 0, 0.1, 0])
    
    # The true plant model is the discretized linear model.
    A_true = A_d
    B_true = B_d
    
    # Initialize the estimated model (for EPSAC predictions)
    A_est_local = A_d.copy()
    B_est_local = B_d.copy()
    
    # Simulation loop: update control every Ts seconds.
    for k in range(N_sim):
        xk = x[:, k]
        # If already settled, return current time as the settling time.
        if np.linalg.norm(xk) < threshold:
            return k * Ts
        # Optimize the control moves using current estimated model.
        U0 = np.zeros(Nc)
        bounds = [(u_min, u_max)] * Nc
        res = minimize(epsac_cost, U0, args=(xk, A_est_local, B_est_local, Nc, Np, C, lam_param),
                       method='SLSQP', bounds=bounds, options={'disp': False})
        U_opt = res.x if res.success else U0
        u_current = U_opt[0]
        
        # Compute predicted next state for adaptation.
        pred_x = A_est_local @ xk + B_est_local.flatten() * u_current
        
        # Simulate the true system.
        x[:, k + 1] = A_true @ xk + B_true.flatten() * u_current
        
        # Update the estimated model using the prediction error.
        error_x = x[:, k + 1] - pred_x
        A_est_local = A_est_local + adaptation_gain * np.outer(error_x, xk)
        B_est_local = B_est_local + adaptation_gain * np.outer(error_x, np.array([u_current])).reshape(B_est_local.shape)
    
    # If the state never settles within T_sim, return a high cost.
    return T_sim + 10

# =============================================================
# 4. GA Optimization Setup Using DEAP
# =============================================================
# Decision vector: 4 parameters [Np, Nc, lam, adaptation_gain]
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Parameter bounds for GA:
toolbox.register("Np_attr", random.uniform, 10, 40)        # Prediction horizon: 10 to 40 steps
toolbox.register("Nc_attr", random.uniform, 1, 10)           # Control horizon: 1 to 10 steps
toolbox.register("lam_attr", random.uniform, 0.01, 1.0)        # Control effort weight: 0.01 to 1.0
toolbox.register("adapt_attr", random.uniform, 0.0001, 0.01)   # Adaptation gain: 0.0001 to 0.01

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.Np_attr, toolbox.Nc_attr, toolbox.lam_attr, toolbox.adapt_attr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation: lower settling time is better.
toolbox.register("evaluate", lambda ind: (simulate_epsac(ind),))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=[10, 1, 0.01, 0.0001], up=[40, 10, 1.0, 0.01], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_epsac_parameters():
    random.seed(42)
    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20,
                                   stats=stats, halloffame=hof, verbose=True)
    best_params = hof[0]
    print("\nOptimized EPSAC Parameters:")
    print("Prediction Horizon (Np):", max(1, int(round(best_params[0]))))
    print("Control Horizon (Nc):", max(1, int(round(best_params[1]))))
    print("Control Effort Weight (lam):", best_params[2])
    print("Adaptation Gain:", best_params[3])
    return best_params

# =============================================================
# 5. Full MuJoCo Simulation Using the Optimized EPSAC Controller
# =============================================================
def run_mujoco_simulation_epsac(opt_params):
    # Unpack and enforce minimum values.
    Np = max(1, int(round(opt_params[0])))
    Nc = max(1, int(round(opt_params[1])))
    lam_param = opt_params[2]
    adaptation_gain = opt_params[3]
    
    # Load the segway model (ensure segway.xml is in your working directory)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions (small disturbance: start with a slight pendulum deviation)
    data.qpos[0] = 0.0      # cart position
    data.qpos[1] = 0.1      # pendulum angle (radians)
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    T_sim = 10.0
    sim_dt = model.opt.timestep  # simulation timestep (e.g., 0.001 s)
    sim_steps = int(T_sim / sim_dt)
    control_interval = int(Ts / sim_dt)  # EPSAC update every Ts seconds
    
    time_data = []
    state_data = []
    control_data = []
    
    # Initialize estimated model for EPSAC.
    A_est_local = A_d.copy()
    B_est_local = B_d.copy()
    
    # Initialize GLFW for visualization.
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "EPSAC MuJoCo Simulation", None, None)
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    def get_state(data):
        return np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    t_sim = 0.0
    step_count = 0
    u_current = 0.0
    while step_count < sim_steps:
        # EPSAC control update at specified intervals.
        if step_count % control_interval == 0:
            xk = get_state(data)
            U0 = np.zeros(Nc)
            bounds = [(u_min, u_max)] * Nc
            res = minimize(epsac_cost, U0, args=(xk, A_est_local, B_est_local, Nc, Np, C, lam_param),
                           method='SLSQP', bounds=bounds, options={'disp': False})
            U_opt = res.x if res.success else U0
            u_current = U_opt[0]
            pred_x = A_est_local @ xk + B_est_local.flatten() * u_current
        data.ctrl[0] = u_current
        mj.mj_step(model, data)
        
        if (step_count + 1) % control_interval == 0:
            x_next = get_state(data)
            error_x = x_next - pred_x
            A_est_local = A_est_local + adaptation_gain * np.outer(error_x, xk)
            B_est_local = B_est_local + adaptation_gain * np.outer(error_x, np.array([u_current])).reshape(B_est_local.shape)
        
        if step_count % control_interval == 0:
            time_data.append(t_sim)
            state_data.append(get_state(data).copy())
            control_data.append(u_current)
        
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt_vis, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        if glfw.window_should_close(window):
            break
        t_sim += sim_dt
        step_count += 1
    
    glfw.terminate()
    
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    # Plot simulation results.
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
    plt.title("EPSAC Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================
# 6. Main: Run GA Optimization Then MuJoCo Simulation with EPSAC
# =============================================================
if __name__ == "__main__":
    best_params = optimize_epsac_parameters()
    run_mujoco_simulation_epsac(best_params)
