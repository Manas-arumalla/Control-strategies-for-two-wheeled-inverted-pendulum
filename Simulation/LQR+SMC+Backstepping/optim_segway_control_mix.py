import numpy as np
import control as ctrl      # pip install control (requires slycot for LQR)
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import random
import mujoco as mj
from mujoco.glfw import glfw
import time

# =============================================================
# 1. Physical and Derived Parameters
# =============================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # wheel moment of inertia [kg*m^2]
Ip = 5.0 * (0.4**2)          # pendulum moment of inertia [kg*m^2]
l = 0.4            # pendulum length [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters for state-space model
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Continuous-time state-space model (states: [base pos, base vel, pend angle, pend ang vel])
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

# =============================================================
# 2. Cost Function for Combined Control Settling Time
# =============================================================
def settling_time_cost_combined(params):
    """
    Candidate decision vector (length 16):
      [Q1, Q2, Q3, Q4, R, lam, eta, eps_smc, k1, k2, k3, k4, alpha, beta, gamma, alpha_filter]
      
    The controller is defined as:
      - LQR: K computed from Q = diag([Q1, Q2, Q3, Q4]) and scalar R.
      - SMC: u_smc = -eta * tanh((x3 + lam*x4) / eps_smc)
      - Backstepping: u_bs = -k1*x3 - k2*x4 - k3*(x1+x2) - k4*(x3+x4)
      - Combined control: u_raw = alpha*u_lqr + beta*u_bs + gamma*u_smc.
      - Filtered control: filtered_u(k+1) = (1 - alpha_filter)*filtered_u(k) + alpha_filter*u_raw.
      
    The plant is simulated with x_dot = A*x + B*u, starting from
      x0 = [0, 0, 0.2, 0].
    Settling time is defined as the first time t at which ||x(t)|| < 0.05 for all future times.
    If settling is not achieved or if the state diverges, a high cost is returned.
    """
    Q1, Q2, Q3, Q4, R_val, lam, eta, eps_smc, k1, k2, k3, k4, alpha_w, beta_w, gamma_w, alpha_filter = params

    # Compute LQR gain
    Q = np.diag([Q1, Q2, Q3, Q4])
    R = np.array([[R_val]])
    try:
        K, _, _ = ctrl.lqr(A, B, Q, R)
        K = np.asarray(K)
    except Exception:
        return 1e6,
    
    T_sim = 5.0
    dt_sim = 0.01
    N = int(T_sim / dt_sim) + 1
    t_vec = np.linspace(0, T_sim, N)
    
    x = np.zeros((4, N))
    x[:, 0] = [0.0, 0.0, 0.2, 0.0]  # initial disturbance in pendulum angle
    filtered_u = 0.0
    
    eps_settle = 0.05
    settling_time = T_sim + 10  # high default cost if not settled
    
    for k in range(N - 1):
        # LQR control
        u_lqr = -np.dot(K, x[:, k])
        # Sliding Mode Control (SMC)
        x3 = x[2, k]
        x4 = x[3, k]
        s_val = x3 + lam * x4
        u_smc = -eta * np.tanh(s_val / eps_smc)
        # Backstepping control
        x1 = x[0, k]
        x2 = x[1, k]
        u_bs = -k1*x3 - k2*x4 - k3*(x1+x2) - k4*(x3+x4)
        # Combined control
        u_raw = alpha_w * u_lqr + beta_w * u_bs + gamma_w * u_smc
        
        filtered_u = (1 - alpha_filter) * filtered_u + alpha_filter * u_raw
        u = filtered_u
        
        x_dot = A @ x[:, k] + B.flatten() * u
        x[:, k+1] = x[:, k] + dt_sim * x_dot
        
        if np.linalg.norm(x[:, k+1]) > 1e6 or np.any(np.isnan(x[:, k+1])):
            return 1e6,
    
    for k in range(N):
        if np.linalg.norm(x[:, k]) < eps_settle:
            if np.all(np.linalg.norm(x[:, k:], axis=0) < eps_settle):
                settling_time = t_vec[k]
                break
    
    return settling_time,

# =============================================================
# 3. GA Setup Using DEAP for Combined Control Tuning
# =============================================================
# Decision vector: 16 parameters.
# Bounds for each parameter.
bounds = [
    (50, 200),       # Q1
    (0.5, 5),        # Q2
    (500, 2000),     # Q3
    (0.5, 5),        # Q4
    (0.005, 0.05),   # R
    (2.0, 5.0),      # lam
    (1, 10),         # eta
    (0.005, 0.05),   # eps_smc
    (5, 10),         # k1
    (8, 15),         # k2
    (3, 8),          # k3
    (8, 15),         # k4
    (0.5, 1.0),      # alpha weight
    (0.0, 0.5),      # beta weight
    (0.0, 0.5),      # gamma weight
    (0.0, 0.1)       # alpha_filter
]
NDIM = len(bounds)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                  toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7,
                  toolbox.attr_8, toolbox.attr_9, toolbox.attr_10, toolbox.attr_11,
                  toolbox.attr_12, toolbox.attr_13, toolbox.attr_14, toolbox.attr_15), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", settling_time_cost_combined)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Custom safe mutation operator to eliminate complex values
def safe_mutate(individual, eta, low, up, indpb):
    try:
        mutated_ind, = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    except TypeError:
        # Convert individual to real numbers and retry
        individual[:] = [float(np.real(x)) for x in individual]
        mutated_ind, = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    # Ensure all mutated values are real numbers
    for i, val in enumerate(mutated_ind):
        mutated_ind[i] = float(np.real(val))
    return mutated_ind,

toolbox.register("mutate", safe_mutate,
                 eta=20, low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_control_parameters():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_params = hof[0]
    print("\nOptimized Combined Controller Parameters:")
    print("LQR Q = diag([{:.2f}, {:.2f}, {:.2f}, {:.2f}]), R = {:.4f}".format(
          best_params[0], best_params[1], best_params[2], best_params[3], best_params[4]))
    print("SMC: lam = {:.2f}, eta = {:.2f}, eps_smc = {:.4f}".format(
          best_params[5], best_params[6], best_params[7]))
    print("Backstepping gains: k1 = {:.2f}, k2 = {:.2f}, k3 = {:.2f}, k4 = {:.2f}".format(
          best_params[8], best_params[9], best_params[10], best_params[11]))
    print("Weights: alpha = {:.2f}, beta = {:.2f}, gamma = {:.2f}".format(
          best_params[12], best_params[13], best_params[14]))
    print("Filter coefficient: alpha_filter = {:.4f}".format(best_params[15]))
    return best_params

# =============================================================
# 4. MuJoCo Simulation with Optimized Combined Controller
# =============================================================
def run_mujoco_simulation(opt_params):
    # Unpack optimized parameters.
    Q_vals = opt_params[0:4]
    R_val = opt_params[4]
    lam = opt_params[5]
    eta = opt_params[6]
    eps_smc = opt_params[7]
    k1, k2, k3, k4 = opt_params[8:12]
    alpha_w, beta_w, gamma_w = opt_params[12:15]
    alpha_filter = opt_params[15]
    
    Q = np.diag(Q_vals)
    R = np.array([[R_val]])
    K, _, _ = ctrl.lqr(A, B, Q, R)
    K = np.asarray(K)
    print("Optimized LQR Gain K:", K)
    
    x_ref = np.zeros((4,))
    
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    data.qpos[0] = 0.0
    data.qpos[1] = 0.1
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    duration = 10.0
    dt = model.opt.timestep
    t_sim = 0.0
    
    time_data = []
    state_data = []   # [base pos, base vel, pend angle, pend ang vel]
    control_data = []
    
    filtered_u = 0.0
    
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "Optimized Combined Control Segway", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    while t_sim < duration:
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        u_lqr = -np.dot(K, (state - x_ref))
        x3 = state[2]
        x4 = state[3]
        s_val = x3 + lam * x4
        u_smc = -eta * np.tanh(s_val / eps_smc)
        x1 = state[0]
        x2 = state[1]
        u_bs = -k1*x3 - k2*x4 - k3*(x1+x2) - k4*(x3+x4)
        
        u_raw = alpha_w * u_lqr + beta_w * u_bs + gamma_w * u_smc
        
        filtered_u = (1 - alpha_filter) * filtered_u + alpha_filter * u_raw
        u = filtered_u
        
        data.ctrl[0] = u.item() if isinstance(u, np.ndarray) else u
        
        mj.mj_step(model, data)
        
        time_data.append(t_sim)
        state_data.append(state.copy())
        control_data.append(u)
        
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt_vis, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        t_sim += dt
        if glfw.window_should_close(window):
            break
    
    glfw.terminate()
    
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_data, state_data[:, 0], label="Base Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Base (Slider) Position")
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
    plt.title("Filtered Combined Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================
# 5. Main: Run GA Optimization then MuJoCo Simulation
# =============================================================
def main():
    best_params = optimize_control_parameters()
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
