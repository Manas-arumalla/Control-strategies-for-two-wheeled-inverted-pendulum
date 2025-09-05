import numpy as np
import control as ctrl
from scipy.integrate import solve_ivp
import pygad
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw

# ----------------------------
# Physical and Derived Parameters (fixed)
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
r = 0.0726         # wheel radius [m]
l = 0.4            # pendulum length [m]
g = 9.81           # gravitational acceleration [m/s^2]
Iw = 0.5 * mw * (r**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (l**2)       # moment of inertia of the pendulum [kg*m^2]

# Derived constants for the model (unchanged)
a = 2 * mw + mp + 2 * Iw/(r**2)
b = mp * l
c = mp * (l**2) + Ip
denom = a * c - b**2
d = mp * g * l  # shorthand for mp*g*l

# ----------------------------
# Simulation settings for the cost function (ODE integration)
# ----------------------------
sim_duration = 10.0   # seconds
sim_tol = 0.05        # tolerance for settling (each state)
sim_t_eval = np.linspace(0, sim_duration, 1000)  # time vector for simulation

# Initial condition for the augmented state: [x, x_dot, theta, theta_dot, z5]
z0 = np.array([0.0, 0.0, 0.2, 0.0, 0.2])


def closed_loop_dynamics(t, z, A_aug, B_aug, K):
    """Closed-loop dynamics: z_dot = (A_aug - B_aug*K)*z."""
    return (A_aug - B_aug @ K) @ z


def compute_settling_time(t, z, tol):
    """
    Compute settling time: first time t after which all states remain
    within the tolerance tol for the remainder of the simulation.
    """
    for i in range(len(t)):
        if np.all(np.abs(z[:, i]) < tol):
            if np.all(np.abs(z[:, i:]) < tol):
                return t[i]
    return None  # did not settle within simulation time


def simulate_system(params):
    """
    Simulate the closed-loop augmented dynamics using candidate parameters.
    Candidate parameters:
      params = [q1, q2, q3, q4, q5, R, alpha, delta]
    Returns the settling time if the system settles, or a high cost otherwise.
    """
    q1, q2, q3, q4, q5, R_val, alpha_val, delta_val = params

    Q_aug = np.diag([q1, q2, q3, q4, q5])
    R = np.array([[R_val]])
    
    # Build augmented system matrices using candidate alpha and delta.
    A_aug = np.array([
        [0,   1,               0,               0,       0],
        [0,   0,      -b*d/denom,               0,       0],
        [0,   0,               0,               1,       0],
        [0,   0,       a*d/denom,               0,       0],
        [0,   0,       alpha_val,             1, -alpha_val]
    ])
    B_aug = np.array([
        [0],
        [c/denom],
        [0],
        [-b/denom],
        [delta_val]
    ])
    
    try:
        K, _, _ = ctrl.lqr(A_aug, B_aug, Q_aug, R)
        K = np.asarray(K)
    except Exception:
        return sim_duration + 10.0  # high cost if LQR fails
    
    sol = solve_ivp(fun=lambda t, z: closed_loop_dynamics(t, z, A_aug, B_aug, K),
                    t_span=(0, sim_duration),
                    y0=z0,
                    t_eval=sim_t_eval)
    if not sol.success:
        return sim_duration + 10.0

    T_settle = compute_settling_time(sol.t, sol.y, sim_tol)
    if T_settle is None:
        return sim_duration + 10.0
    else:
        return T_settle


# ----------------------------
# Define the fitness function for GA (must accept 3 parameters)
# ----------------------------
def fitness_func(ga_instance, solution, solution_idx):
    T_settle = simulate_system(solution)
    # Lower settling time gives higher fitness; return negative settling time.
    fitness = -T_settle
    return fitness


# ----------------------------
# GA Parameter Setup (we optimize 8 parameters)
# ----------------------------
gene_space = [
    {'low': 1, 'high': 1000},    # q1: weight on x
    {'low': 0.1, 'high': 100},     # q2
    {'low': 1, 'high': 1000},    # q3: weight on theta
    {'low': 0.1, 'high': 100},     # q4
    {'low': 0.1, 'high': 100},     # q5: penalty for auxiliary state
    {'low': 0.01, 'high': 10},     # R (control penalty)
    {'low': 0.1, 'high': 10},      # alpha (auxiliary dynamics parameter)
    {'low': 0.001, 'high': 1}      # delta (coupling parameter)
]

ga_instance = pygad.GA(num_generations=30,
                       num_parents_mating=4,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=8,
                       gene_space=gene_space,
                       mutation_percent_genes=20,
                       mutation_type="random",
                       crossover_type="single_point",
                       stop_criteria=["reach_0"])


# ----------------------------
# MuJoCo Simulation with Optimized Controller
# ----------------------------
def run_mujoco_simulation(opt_params):
    """
    Use the optimized parameters to compute the LQR gain for the augmented model
    and run a MuJoCo simulation.
    """
    q1, q2, q3, q4, q5, R_val, alpha_val, delta_val = opt_params
    Q_aug = np.diag([q1, q2, q3, q4, q5])
    R = np.array([[R_val]])
    A_aug = np.array([
        [0,   1,               0,               0,       0],
        [0,   0,      -b*d/denom,               0,       0],
        [0,   0,               0,               1,       0],
        [0,   0,       a*d/denom,               0,       0],
        [0,   0,       alpha_val,             1, -alpha_val]
    ])
    B_aug = np.array([
        [0],
        [c/denom],
        [0],
        [-b/denom],
        [delta_val]
    ])
    try:
        K, _, _ = ctrl.lqr(A_aug, B_aug, Q_aug, R)
        K = np.asarray(K)
    except Exception as e:
        print("LQR failed with optimized parameters.")
        return

    print("Optimized Augmented LQR Gain K:", K)
    
    # Load the MuJoCo model.
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial physical states.
    data.qpos[0] = 0.0   # base position
    data.qpos[1] = 0.2   # pendulum angle (radians)
    data.qvel[0] = 0.0   # base velocity
    data.qvel[1] = 0.0   # pendulum angular velocity
    
    duration = 10.0  # simulation duration in seconds
    dt = model.opt.timestep
    t_sim = 0.0

    # Data recording lists.
    time_data = []
    state_data = []   # [x, x_dot, theta, theta_dot]
    control_data = []
    
    # Initialize MuJoCo visualization (GLFW)
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    window = glfw.create_window(1200, 900, "Optimized Augmented Controller Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # Auxiliary state initialization (for augmented model)
    z5 = data.qpos[1].item()  # initialize auxiliary state as the pendulum angle

    # Simulation loop.
    while t_sim < duration:
        # Extract current physical states (convert to scalar).
        x = data.qpos[0].item()
        theta = data.qpos[1].item()
        x_dot = data.qvel[0].item()
        theta_dot = data.qvel[1].item()
        
        # Build augmented state vector: [x, x_dot, theta, theta_dot, z5]
        z = np.array([x, x_dot, theta, theta_dot, z5])
        
        # Compute control input: u = -K * z.
        u = -np.dot(K, z).item()
        
        # Apply control input.
        data.ctrl[0] = u
        
        # Update auxiliary state z5 using Euler integration:
        # z5_dot = theta_dot + alpha*(theta - z5) + delta*u.
        z5_dot = theta_dot + alpha_val * (theta - z5) + delta_val * u
        z5 = z5 + dt * z5_dot
        
        # Step the simulation.
        mj.mj_step(model, data)
        
        # Record simulation data.
        time_data.append(t_sim)
        state_data.append([x, x_dot, theta, theta_dot])
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
    
    # Plot simulation results.
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    control_data = np.array(control_data)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_data, state_data[:, 0], 'b', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Base Position (m)")
    plt.title("Base Position")
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
    plt.title("Control Input")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main: Run GA Optimization then MuJoCo Simulation
# ----------------------------
def main():
    # Run GA to optimize parameters (only once).
    ga_instance.run()
    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
    print("Optimized Parameters:")
    print("q1, q2, q3, q4, q5, R, alpha, delta =")
    print(best_solution)
    print("Best fitness (=-settling time):", best_fitness)
    print("Estimated settling time:", -best_fitness, "s")
    
    # Run the MuJoCo simulation using the best parameters.
    run_mujoco_simulation(best_solution)

if __name__ == "__main__":
    main()
