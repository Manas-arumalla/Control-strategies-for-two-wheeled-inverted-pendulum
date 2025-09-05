import numpy as np
import control as ctrl      # pip install control (requires slycot for hinfsyn)
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms

# =============================================================================
# 1. Plant and Weighting Definition
# =============================================================================
# Plant parameters:
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # wheel moment of inertia [kg*m^2]
Ip = 5.0 * (0.4**2)          # pendulum moment of inertia [kg*m^2]
l  = 0.4           # pendulum length (distance from wheel axle to COM) [m]
r  = 0.0726        # wheel radius [m]
g  = 9.81          # gravitational acceleration [m/s^2]

# Derived parameters (for the linearized model):
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Construct the state-space model (states: [cart pos, cart vel, pend angle, pend ang vel])
A = np.array([[0,    1,      0,        0],
              [0,    0,  -b*d/Delta,    0],
              [0,    0,      0,        1],
              [0,    0,   a*d/Delta,    0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])
# Measured outputs: cart position and pendulum angle
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],
              [0]])

# Create the plant
sys = ctrl.ss(A, B, C, D)

# Define weighting functions (convert to transfer functions)
s = ctrl.tf('s')
# The decision variables will define:
#   Wp(s) = (s + p1)/(s + p2) and Wu(s) = p3/1.
# Here we set default values for now.
Wp = (s + 10) / (s + 0.1)   
Wu = ctrl.tf(0.1, 1)

# Form the augmented (generalized) plant
P = ctrl.augw(sys, Wp, None, Wu)

# =============================================================================
# 2. H∞ Synthesis in Python
# =============================================================================
nmeas = 2  # measured outputs (cart pos and pend angle)
ncon = 1   # one control input

# Synthesize the H∞ controller using hinfsyn.
K_hinf, CL, gamma, info = ctrl.hinfsyn(P, nmeas, ncon)

print("Achieved H performance gamma:", gamma)
print("H controller K:")
print(K_hinf)

# Extract state-space matrices of the controller
A_hinf, B_hinf, C_hinf, D_hinf = ctrl.ssdata(K_hinf)

# Initialize controller state
xK = np.zeros((A_hinf.shape[0],))

# =============================================================================
# 3. GA Cost Function for H∞ Weighting Tuning
# =============================================================================
# We want to optimize the weighting functions used in H∞ synthesis.
# Decision vector: [p1, p2, p3]
# where Wp(s) = (s + p1)/(s + p2) and Wu(s) = p3/1.
# Suggested bounds:
#   p1 in [1, 100]
#   p2 in [0.01, 10]
#   p3 in [0.001, 1]

def hinf_ga_cost(params):
    p1, p2, p3 = params

    try:
        # Define weighting functions using candidate parameters
        s = ctrl.tf('s')
        Wp = (s + p1) / (s + p2)
        Wu = ctrl.tf(p3, 1)
        # Form the augmented plant
        P_aug = ctrl.augw(sys, Wp, None, Wu)
        # Synthesize the H∞ controller
        nmeas = 2
        ncon = 1
        K_hinf, CL, gamma, info = ctrl.hinfsyn(P_aug, nmeas, ncon)
    except Exception as e:
        return 1e6,
    
    try:
        A_h, B_h, C_h, D_h = ctrl.ssdata(K_hinf)
    except Exception as e:
        return 1e6,
    
    # Simulate the closed-loop system (plant + controller) using Euler integration.
    T_sim = 10.0
    dt_sim = 0.001
    N = int(T_sim / dt_sim) + 1
    x = np.zeros((4, N))
    xK = np.zeros((A_h.shape[0], N))
    # initial condition: small pendulum angle deviation
    x[:, 0] = [0, 0, 0.1, 0]
    cost = 0.0
    max_state_norm = 1e6
    
    for k in range(N - 1):
        # Measured output: cart position and pendulum angle
        y = np.array([x[0, k], x[2, k]])
        # Update controller state (Euler integration)
        xK_dot = A_h.dot(xK[:, k]) + B_h.dot(y)
        xK[:, k+1] = xK[:, k] + dt_sim * xK_dot
        # Compute control input: u = C_h*xK + D_h*y
        u = C_h.dot(xK[:, k]) + D_h.dot(y)
        # Update plant state
        x_dot = A.dot(x[:, k]) + B.flatten() * u
        x[:, k+1] = x[:, k] + dt_sim * x_dot
        # Accumulate cost (squared error)
        cost += np.sum(y**2) * dt_sim
        if np.linalg.norm(x[:, k+1]) > max_state_norm or np.any(np.isnan(x[:, k+1])):
            return 1e6,
    # Optionally add the achieved gamma to the cost (weighted)
    alpha = 1.0
    cost += alpha * gamma
    return cost,

# =============================================================================
# 4. GA Setup Using DEAP for H∞ Weighting Tuning
# =============================================================================
NDIM = 3

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def random_param(low, high):
    return random.uniform(low, high)

bounds = [(1, 100), (0.01, 10), (0.001, 1)]
for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", hinf_ga_cost)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Define a safe mutation operator that ensures real outputs.
def safe_mutate(individual, eta, low, up, indpb):
    # Use the default polynomial bounded mutation
    mutated_ind, = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    # Replace any complex numbers with their real part
    for i, val in enumerate(mutated_ind):
        if isinstance(val, complex):
            mutated_ind[i] = val.real
    return mutated_ind,

toolbox.register("mutate", safe_mutate,
                 eta=20, low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_weights():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_params = hof[0]
    print("\nOptimized Weighting Parameters:")
    print(f"p1 (Wp numerator offset) = {best_params[0]:.2f}")
    print(f"p2 (Wp denominator offset) = {best_params[1]:.2f}")
    print(f"p3 (Wu gain) = {best_params[2]:.4f}")
    return best_params

# =============================================================================
# 5. MuJoCo Simulation Setup with the Optimized H∞ Controller
# =============================================================================
def run_mujoco_simulation(opt_params):
    p1, p2, p3 = opt_params
    s = ctrl.tf('s')
    Wp = (s + p1) / (s + p2)
    Wu = ctrl.tf(p3, 1)
    P_aug = ctrl.augw(sys, Wp, None, Wu)
    nmeas = 2
    ncon = 1
    try:
        K_hinf, CL, gamma, info = ctrl.hinfsyn(P_aug, nmeas, ncon)
    except Exception as e:
        print("H∞ synthesis failed with optimized weights!")
        return
    print("Optimized H∞ gamma:", gamma)
    print("Optimized H∞ controller K:")
    print(K_hinf)
    
    A_h, B_h, C_h, D_h = ctrl.ssdata(K_hinf)
    xK = np.zeros((A_h.shape[0],))
    
    # Load the MuJoCo model (ensure "segway.xml" is in your working directory)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions (small initial angle)
    data.qpos[0] = 0.0    # cart position [m]
    data.qpos[1] = 0.1    # pendulum angle [rad]
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    duration = 10.0
    dt = model.opt.timestep
    t_sim = 0.0
    
    time_data = []
    state_data = []   # [cart pos, cart vel, pend angle, pend ang vel]
    control_data = []
    
    if not glfw.init():
        print("GLFW initialization failed")
        return
    window = glfw.create_window(1200, 900, "Optimized H∞ Controlled Segway", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window creation failed")
        return
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt_vis = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    while t_sim < duration:
        y = np.array([data.qpos[0], data.qpos[1]])
        xK_dot = A_h.dot(xK) + B_h.dot(y)
        xK = xK + dt * xK_dot
        u = C_h.dot(xK) + D_h.dot(y)
        data.ctrl[0] = u.item()
        mj.mj_step(model, data)
        
        time_data.append(t_sim)
        state_data.append([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        control_data.append(u.item())
        
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
    plt.plot(time_data, state_data[:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart Position")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_data, state_data[:, 2], color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")
    plt.title("Pendulum Angle")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_data, control_data, color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (N)")
    plt.title("Control Signal")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. Main: Optimize and Run Simulation
# =============================================================================
def main():
    best_params = optimize_weights()
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
