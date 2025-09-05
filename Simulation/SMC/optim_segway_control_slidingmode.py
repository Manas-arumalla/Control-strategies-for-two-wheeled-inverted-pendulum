import numpy as np
import matplotlib.pyplot as plt
import pygad
import mujoco as mj
from mujoco.glfw import glfw
import control as ctrl  # only used if needed elsewhere; not directly needed for SMC

# =============================================================================
# 1. Physical Parameters and Derived Quantities
# =============================================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
r = 0.0726         # wheel radius [m]
l = 0.4            # pendulum length [m]
g = 9.81           # gravitational acceleration [m/s^2]
Iw = 0.5 * mw * (r**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (l**2)       # moment of inertia of the pendulum [kg*m^2]

# Derived constants (from your MATLAB derivation)
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Linearized state-space model (for cost function simulation)
# States: [base position, base velocity, pendulum angle, pendulum angular velocity]
A = np.array([[0, 1, 0, 0],
              [0, 0, -b*d/Delta, 0],
              [0, 0, 0, 1],
              [0, 0, a*d/Delta, 0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

# =============================================================================
# 2. SMC Law Definitions
# =============================================================================
def sat(value):
    """Saturation function limiting value to [-1, 1]."""
    return np.clip(value, -1.0, 1.0)

def compute_smc_control(x, smc_params):
    """
    Compute the SMC control input given state x and SMC parameters.
    
    smc_params: [lambda1, lambda2, lambda3, K_smc, phi]
      - lambda1, lambda2, lambda3: sliding surface gains.
      - K_smc: sliding mode gain.
      - phi: boundary layer thickness.
    """
    lambda1, lambda2, lambda3, K_smc, phi = smc_params
    
    # Define sliding surface s (linear combination of states)
    s = lambda1 * x[0] + x[1] + lambda2 * x[2] + lambda3 * x[3]
    s_sat = sat(s / phi)
    
    # Compute an equivalent derivative term (this is one design choice)
    s_dot_equiv = lambda1 * x[1] + lambda2 * x[3] + (d / Delta) * (-b + lambda3 * a) * x[2]
    # Gain factor (ensuring proper units)
    k_u = (c - lambda3 * b) / Delta
    
    # Sliding mode control law:
    u = (1.0 / k_u) * (-s_dot_equiv - K_smc * s_sat)
    return u

# =============================================================================
# 3. Cost Function for SMC Tuning (using ODE integration)
# =============================================================================
def settling_time_cost_smc(params):
    """
    Evaluate candidate SMC parameters by simulating the closed-loop dynamics:
        x_dot = A*x + B*u,  u = compute_smc_control(x, smc_params)
    Candidate parameters (length 5):
      [lambda1, lambda2, lambda3, K_smc, phi]
    
    The initial state is set with a disturbance in the pendulum angle.
    Settling time is defined as the first time when ||x|| < eps_settle and remains
    below eps_settle. If the state diverges, a high cost is returned.
    """
    # Unpack candidate parameters
    lambda1, lambda2, lambda3, K_smc, phi = params
    smc_params = [lambda1, lambda2, lambda3, K_smc, phi]
    
    # Simulation settings
    T_sim = 5.0      # total simulation time (seconds)
    dt_sim = 0.01    # integration timestep (seconds)
    N = int(T_sim / dt_sim) + 1
    t_vec = np.linspace(0, T_sim, N)
    
    # Initial state (small disturbance in pendulum angle)
    x = np.zeros((4, N))
    x[:, 0] = [0.0, 0.0, 0.1, 0.0]
    
    eps_settle = 0.05  # threshold for settling
    settling_time = T_sim + 10  # default high cost if not settled
    
    # Euler integration of the dynamics:
    for k in range(N - 1):
        u = compute_smc_control(x[:, k], smc_params)
        x_dot = A @ x[:, k] + B.flatten() * u
        x[:, k+1] = x[:, k] + dt_sim * x_dot
        
        # Check for divergence
        if np.linalg.norm(x[:, k+1]) > 1e6 or np.any(np.isnan(x[:, k+1])):
            return T_sim + 10,
    
    # Determine settling time:
    for k in range(N):
        if np.linalg.norm(x[:, k]) < eps_settle:
            if np.all(np.linalg.norm(x[:, k:], axis=0) < eps_settle):
                settling_time = t_vec[k]
                break
                
    return settling_time,

# =============================================================================
# 4. GA Optimization using PyGAD for SMC Parameters
# =============================================================================
# We optimize 5 parameters: [lambda1, lambda2, lambda3, K_smc, phi]
# Define the gene space (bounds) for each parameter.
gene_space = [
    {'low': 1.0, 'high': 10.0},    # lambda1
    {'low': 1.0, 'high': 20.0},    # lambda2
    {'low': 0.1, 'high': 10.0},    # lambda3
    {'low': 1.0, 'high': 50.0},    # K_smc
    {'low': 0.001, 'high': 0.1}    # phi
]

def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for PyGAD.
    We simulate the system using candidate SMC parameters and return negative settling time.
    """
    T_settle = settling_time_cost_smc(solution)[0]
    fitness = -T_settle  # lower settling time => higher fitness
    return fitness

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=10,
                       fitness_func=fitness_func,
                       sol_per_pop=50,
                       num_genes=5,
                       gene_space=gene_space,
                       mutation_percent_genes=20,
                       mutation_type="random",
                       crossover_type="single_point",
                       stop_criteria=["reach_0"])

# =============================================================================
# 5. MuJoCo Simulation with Optimized SMC Controller
# =============================================================================
def run_mujoco_simulation(smc_params):
    """
    Run a MuJoCo simulation using the optimized SMC parameters.
    smc_params: [lambda1, lambda2, lambda3, K_smc, phi]
    """
    # Load the MuJoCo model (ensure "segway.xml" is available)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions: small disturbance in pendulum angle.
    data.qpos[0] = 0.0   # base position
    data.qpos[1] = 0.1   # pendulum angle (radians)
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0

    duration = 10.0
    dt = model.opt.timestep
    t_sim = 0.0

    # Data storage for plotting
    time_data = []
    state_data = []   # [base pos, base vel, pend angle, pend ang vel]
    control_data = []
    
    # Initialize MuJoCo visualization (GLFW)
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "Optimized SMC Segway Simulation", None, None)
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
        # Get current state: [base pos, base vel, pend angle, pend ang vel]
        current_state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        # Compute SMC control input using optimized parameters
        u_current = compute_smc_control(current_state, smc_params)
        data.ctrl[0] = u_current
        
        # Step simulation
        mj.mj_step(model, data)
        
        # Record data
        time_data.append(t_sim)
        state_data.append(current_state.copy())
        control_data.append(u_current)
        
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
    
    # Convert data to arrays and plot
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_data, state_data[:, 0], 'b', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Base Position (m)")
    plt.title("Base (Slider) Position")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_data, state_data[:, 2], 'r', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_data, control_data, 'g', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (N)")
    plt.title("SMC Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. Main: Run GA Optimization then MuJoCo Simulation
# =============================================================================
def main():
    print("Optimizing SMC parameters using PyGAD...")
    ga_instance.run()
    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
    print("Optimized SMC Parameters:")
    print("lambda1, lambda2, lambda3, K_smc, phi =")
    print(best_solution)
    print("Best fitness (=-settling time):", best_fitness)
    print("Estimated settling time:", -best_fitness, "s")
    
    print("\nRunning MuJoCo simulation with optimized SMC controller...")
    run_mujoco_simulation(best_solution)

if __name__ == "__main__":
    main()
