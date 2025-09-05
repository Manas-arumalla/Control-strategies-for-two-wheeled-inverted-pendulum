import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import pygad
import mujoco as mj
from mujoco.glfw import glfw
import time
import random

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
Ts = 0.05  # Sampling time (seconds)
sysd = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), Ts)
A_d = sysd[0]
B_d = sysd[1]

# Terminal cost from discrete ARE
Q_state = np.diag([100, 1, 1000, 1])
R_state = np.array([[0.01]])
P_terminal = solve_discrete_are(A_d, B_d, Q_state, R_state)

# =============================================================
# 3. MPC Tuning Parameters to Optimize via GA
# =============================================================
# Candidate parameters (in order):
#   W1: weight for cart position error [50, 200]
#   W2: weight for pendulum angle error [500, 2000]
#   R_weight: weight on control effort [0.001, 0.1]
#   Delta_u_weight: weight on change in control input [0.01, 1]
#   N: Prediction horizon (steps) [10, 30] (will be rounded to an integer)
#   Nu: Control horizon (steps) [1, 10] (must satisfy Nu <= N)
bounds = [
    (50, 200),       # W1
    (500, 2000),     # W2
    (0.001, 0.1),    # R_weight
    (0.01, 1),       # Delta_u_weight
    (10, 30),        # N (prediction horizon)
    (1, 10)          # Nu (control horizon)
]

# =============================================================
# 4. MPC Solver using CVXPY with Constant Wrapping
# =============================================================
def solve_mpc_tuned(x0, u_prev, ref, A_d, B_d, N, Nu, P_terminal, W_out, R_weight, Delta_u_weight):
    n = A_d.shape[0]
    m = B_d.shape[1]

    
    # Wrap candidate parameters as constants
    W_out_const = cp.Constant(W_out)
    R_weight_const = cp.Constant(R_weight)
    Delta_u_weight_const = cp.Constant(Delta_u_weight)
    ref_const = cp.Constant(ref)
    
    # Decision variables for state trajectory and control inputs
    x = cp.Variable((n, N+1))
    u = cp.Variable((m, N))
    
    cost = 0
    constraints = [x[:, 0] == x0]
    
    for k in range(N):
        constraints += [x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k]]
        constraints += [u[:, k] >= -10, u[:, k] <= 10]
        if k >= Nu:
            constraints += [u[:, k] == u[:, Nu-1]]
            
        y_k = C @ x[:, k]
        cost += cp.quad_form(y_k - ref_const, W_out_const)
        cost += R_weight_const * cp.sum_squares(u[:, k])
        if k == 0:
            cost += Delta_u_weight_const * cp.sum_squares(u[:, k] - u_prev)
        else:
            cost += Delta_u_weight_const * cp.sum_squares(u[:, k] - u[:, k-1])
    
    cost += cp.quad_form(x[:, N], cp.Constant(P_terminal))
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        # Try solving with OSQP without warm-start, with relaxed tolerances
        prob.solve(solver=cp.OSQP, warm_start=False, verbose=True,
                   max_iter=50000, eps_abs=1e-4, eps_rel=1e-4)
    except cp.SolverError as e:
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
    # Unpack candidate parameters and round integer ones for horizon lengths
    W1, W2, R_weight, Delta_u_weight, N_cont, Nu_cont = params
    N = int(round(N_cont))
    Nu = int(round(Nu_cont))
    if Nu > N:
        Nu = N
    W_out = np.diag([W1, W2])
    
    T_sim = 5.0
    dt_sim = Ts
    N_steps = int(T_sim / dt_sim) + 1
    t_vec = np.linspace(0, T_sim, N_steps)
    
    # Initialize state and previous control input
    x = np.zeros((4, N_steps))
    x[:, 0] = [0.0, 0.0, 0.1, 0.0]
    u_prev = np.array([0.0])
    
    eps_settle = 0.05
    settling_time = T_sim + 10  # default high cost if not settled
    
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
                
    return settling_time

# =============================================================
# 6. Define the Fitness Function for PyGAD
# =============================================================
def fitness_func(ga_instance, solution, solution_idx):
    T_settle = mpc_settling_time_cost(solution)
    # We want to minimize settling time. Since PyGAD maximizes fitness,
    # we return the negative of the settling time.
    fitness = -T_settle
    return fitness

gene_space = [
    {'low': 50, 'high': 200},       # W1
    {'low': 500, 'high': 2000},     # W2
    {'low': 0.001, 'high': 0.1},    # R_weight
    {'low': 0.01, 'high': 1},       # Delta_u_weight
    {'low': 10, 'high': 30},        # N (prediction horizon)
    {'low': 1, 'high': 10}          # Nu (control horizon)
]

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
    
    # Set initial conditions for the simulation
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
    window = glfw.create_window(1200, 900, "Tuned MPC Segway (PyGAD)", None, None)
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
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=10,
                           fitness_func=fitness_func,
                           sol_per_pop=50,
                           num_genes=6,
                           gene_space=gene_space,
                           mutation_probability=0.2,
                           mutation_type="random",
                           crossover_type="single_point",
                           stop_criteria=["reach_0"])
    
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print("\nOptimized MPC Parameters:")
    print(f"W_out = diag([{best_solution[0]:.2f}, {best_solution[1]:.2f}])")
    print(f"R_weight = {best_solution[2]:.4f}, Delta_u_weight = {best_solution[3]:.4f}")
    print(f"N (prediction horizon) = {int(round(best_solution[4]))}, Nu (control horizon) = {int(round(best_solution[5]))}")
    print(f"Best fitness (=-settling time): {best_solution_fitness}")
    print(f"Estimated settling time: {-best_solution_fitness:.4f} s")
    
    run_mujoco_simulation(best_solution)

if __name__ == "__main__":
    main()
