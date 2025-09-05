import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import mujoco as mj
from mujoco.glfw import glfw
import time

# =============================================================
# 1. Physical Parameters and Continuous-Time Model
# =============================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # wheel moment of inertia [kg*m^2]
Ip = 5.0 * (0.4**2)          # pendulum moment of inertia [kg*m^2]
l = 0.4            # pendulum length [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Continuous-time state-space model
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

# =============================================================
# 2. Discretize the System for MPC
# =============================================================
Ts = 0.05  # MPC sampling time (seconds)
sysd = cont2discrete((A, B, np.eye(4), np.zeros((4,1))), Ts)
A_d = sysd[0]
B_d = sysd[1]

# Terminal cost from discrete ARE
Q_state = np.diag([100, 1, 1000, 1])
R_state = np.array([[0.01]])
P_terminal = solve_discrete_are(A_d, B_d, Q_state, R_state)

# =============================================================
# 3. MPC Tuning Parameters to Optimize via GA
# =============================================================
# We tune:
#   W1: weight for cart position error in output cost [50, 200]
#   W2: weight for pendulum angle error [500, 2000]
#   R_weight: weight on control effort [0.001, 0.1]
#   Delta_u_weight: weight on change in control input [0.01, 1]
#   N: Prediction horizon (steps) [10, 30] (will be rounded to int)
#   Nu: Control horizon (steps) [1, 10] (must satisfy Nu <= N)
bounds = [
    (50, 200),        # W1
    (500, 2000),      # W2
    (0.001, 0.1),     # R_weight
    (0.01, 1),        # Delta_u_weight
    (10, 30),         # N
    (1, 10)           # Nu
]

# =============================================================
# 4. MPC Solver using CVXPY with Tighter Settings and Constant Wrapping
# =============================================================
def solve_mpc_tuned(x0, u_prev, ref, A_d, B_d, N, Nu, P_terminal, W_out, R_weight, Delta_u_weight):
    n = A_d.shape[0]
    m = B_d.shape[1]
    
    # Wrap candidate parameters as constants to ensure DCP compliance
    W_out_const = cp.Constant(W_out)
    R_weight_const = cp.Constant(R_weight)
    Delta_u_weight_const = cp.Constant(Delta_u_weight)
    ref_const = cp.Constant(ref)
    
    # Decision variables for state trajectory and control inputs:
    x = cp.Variable((n, N+1))
    u = cp.Variable((m, N))
    
    cost = 0
    constraints = []
    constraints += [x[:, 0] == x0]
    
    for k in range(N):
        constraints += [x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k]]
        constraints += [u[:, k] >= -10, u[:, k] <= 10]
        if k >= Nu:
            constraints += [u[:, k] == u[:, Nu-1]]
        
        y_k = C @ x[:, k]
        cost += cp.quad_form(y_k - ref_const, W_out_const)
        cost += R_weight_const * cp.sum_squares(u[:, k])
        if k == 0:
            # Wrap u_prev as constant
            cost += Delta_u_weight_const * cp.sum_squares(u[:, k] - cp.Constant(u_prev))
        else:
            cost += Delta_u_weight_const * cp.sum_squares(u[:, k] - u[:, k-1])
    
    cost += cp.quad_form(x[:, N], cp.Constant(P_terminal))
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=True,
                   max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
    except cp.SolverError:
        print("OSQP failed, switching to ECOS...")
        prob.solve(solver=cp.ECOS, verbose=True)
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return u[:, 0].value.item()
    else:
        print("MPC optimization failed. Status:", prob.status)
        return 0.0

# =============================================================
# 5. MPC Cost Function for GA: Compute Settling Time
# =============================================================
def mpc_settling_time_cost(params):
    # Unpack candidate parameters and round integer ones
    W1, W2, R_weight, Delta_u_weight, N_cont, Nu_cont = params
    N = int(round(N_cont))
    Nu = int(round(Nu_cont))
    if Nu > N:
        Nu = N
    W_out = np.diag([W1, W2])
    
    T_sim = 5.0
    dt_sim = Ts
    N_steps = int(T_sim/dt_sim) + 1
    t_vec = np.linspace(0, T_sim, N_steps)
    
    x = np.zeros((4, N_steps))
    x[:, 0] = [0.0, 0.0, 0.1, 0.0]
    u_prev = np.array([0.0])
    
    eps_settle = 0.05
    settling_time = T_sim + 10  # high default if not settled
    
    for k in range(N_steps - 1):
        ref = np.array([0.0, 0.0])
        u_current = solve_mpc_tuned(x[:, k], u_prev, ref, A_d, B_d, N, Nu, P_terminal, W_out, R_weight, Delta_u_weight)
        u_prev = np.array([u_current])
        x[:, k+1] = A_d @ x[:, k] + B_d.flatten() * u_current
        
    for k in range(N_steps):
        if np.linalg.norm(x[:, k]) < eps_settle:
            if np.all(np.linalg.norm(x[:, k:], axis=0) < eps_settle):
                settling_time = t_vec[k]
                break
                
    return settling_time,

# =============================================================
# 6. GA Setup Using DEAP for MPC Parameter Tuning
# =============================================================
NDIM = 6
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                  toolbox.attr_4, toolbox.attr_5), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", mpc_settling_time_cost)
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

def optimize_mpc_parameters():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_params = hof[0]
    print("\nOptimized MPC Parameters:")
    print(f"W_out = diag([{best_params[0]:.2f}, {best_params[1]:.2f}])")
    print(f"R_weight = {best_params[2]:.4f}, Delta_u_weight = {best_params[3]:.4f}")
    print(f"N (prediction horizon) = {int(round(best_params[4]))}, Nu (control horizon) = {int(round(best_params[5]))}")
    return best_params

# =============================================================
# 7. MuJoCo Simulation with Optimized MPC Controller
# =============================================================
def run_mujoco_simulation(opt_params):
    W1, W2, R_weight, Delta_u_weight, N_cont, Nu_cont = opt_params
    N = int(round(N_cont))
    Nu = int(round(Nu_cont))
    if Nu > N:
        Nu = N
    W_out = np.diag([W1, W2])
    
    u_prev = np.array([0.0])
    
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    data.qpos[0] = 0.0
    data.qpos[1] = 0.1
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    duration = 10.0
    dt_sim = model.opt.timestep
    sim_steps = int(duration / dt_sim)
    
    time_data = []
    state_data = []
    control_data = []
    
    mpc_update_steps = int(Ts / dt_sim)
    
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "Tuned MPC Segway", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    step_count = 0
    t_sim = 0.0
    u_current = 0.0
    while step_count < sim_steps:
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        ref = np.array([0.0, 0.0])
        if step_count % mpc_update_steps == 0:
            u_current = solve_mpc_tuned(state, u_prev, ref, A_d, B_d, N, Nu, P_terminal, W_out, R_weight, Delta_u_weight)
            u_prev = np.array([u_current])
        
        data.ctrl[0] = u_current
        mj.mj_step(model, data)
        
        time_data.append(t_sim)
        state_data.append(state.copy())
        control_data.append(u_current)
        
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt_vis, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        t_sim += dt_sim
        step_count += 1
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
    plt.title("MPC Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================
# 8. Main: Run GA Optimization then MuJoCo Simulation
# =============================================================
def main():
    best_params = optimize_mpc_parameters()
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
