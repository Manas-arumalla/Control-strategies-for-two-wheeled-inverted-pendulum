import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import control as ctrl      # pip install control
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import time

# =============================================================
# 1. Physical Parameters and Continuous-Time Model
# =============================================================
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # wheel inertia [kg*m^2]
Ip = 5.0 * (0.4**2)          # pendulum inertia [kg*m^2]
l = 0.4            # pendulum length [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters (from the linearized model)
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Continuous-time state-space model
# State vector: x = [base position, base velocity, pendulum angle, pendulum angular velocity]^T
A = np.array([[0, 1, 0, 0],
              [0, 0, -b*d/Delta, 0],
              [0, 0, 0, 1],
              [0, 0, a*d/Delta, 0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

x_ref = np.zeros(4)  # target state is zero

# =============================================================
# 2. Closed-Loop Simulation and Cost Function
# =============================================================
def simulate_closed_loop(desired_poles_candidate):
    """
    Given a candidate vector of 4 positive numbers, the actual desired closed-loop poles are:
         poles = -[p1, p2, p3, p4].
    Compute the state-feedback gain via pole placement, then simulate the closed-loop
    system (Euler integration) from an initial disturbance and return the settling time.
    If the state diverges, return a high cost.
    """
    poles = -np.array(desired_poles_candidate)
    try:
        K = ctrl.place(A, B, poles)
        K = np.asarray(K)
    except Exception:
        return 1e6  # high cost if pole placement fails
    
    A_cl = A - B @ K
    T_sim = 10.0
    dt = 0.001
    N = int(T_sim/dt) + 1
    t_vec = np.linspace(0, T_sim, N)
    x = np.zeros((4, N))
    x[:, 0] = [0, 0, 0.1, 0]  # initial disturbance (small pendulum angle)
    
    for k in range(N-1):
        x[:, k+1] = x[:, k] + dt * (A_cl @ x[:, k])
        if np.linalg.norm(x[:, k+1]) > 1e6 or np.any(np.isnan(x[:, k+1])):
            return 1e6  # divergence penalty
    
    eps = 0.05
    settling_time = T_sim + 10
    for k in range(N):
        if np.linalg.norm(x[:, k]) < eps:
            if np.all(np.linalg.norm(x[:, k:], axis=0) < eps):
                settling_time = t_vec[k]
                break
    return settling_time

def cost_function(individual):
    return simulate_closed_loop(individual),

# =============================================================
# 3. GA Setup with DEAP for Pole Placement Tuning
# =============================================================
NDIM = 4
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Candidate parameters between 0.1 and 10
bounds = [(0.1, 10)] * NDIM

def random_param(low, high):
    return random.uniform(low, high)

for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", random_param, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", cost_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Custom safe mutation operator using uniform (Gaussian) perturbation
def safe_mutate_uniform(individual, sigma=0.1, indpb=0.2, low=None, up=None):
    for i in range(len(individual)):
        if random.random() < indpb:
            # Add Gaussian noise
            individual[i] += random.gauss(0, sigma)
            # Clip to bounds
            if low is not None and up is not None:
                individual[i] = max(low[i], min(up[i], individual[i]))
    return individual,

toolbox.register("mutate", safe_mutate_uniform,
                 sigma=0.1, indpb=0.2, low=[b[0] for b in bounds], up=[b[1] for b in bounds])
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_controller():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_ind = hof[0]
    print("\nOptimized Desired Pole Magnitudes (positive values):", best_ind)
    print("Resulting Closed-loop Poles:", -np.array(best_ind))
    return best_ind

# =============================================================
# 4. MuJoCo Simulation with Optimized Pole Placement Controller
# =============================================================
def run_mujoco_simulation(desired_poles_candidate):
    desired_poles = -np.array(desired_poles_candidate)
    try:
        K = ctrl.place(A, B, desired_poles)
        K = np.asarray(K)
    except Exception as e:
        print("Pole placement failed with candidate:", desired_poles_candidate)
        return
    print("Optimized Pole Placement Gain K:", K)
    
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Set initial conditions (disturbance in pendulum angle)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.1
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    
    duration = 10.0
    dt = model.opt.timestep
    t_sim = 0.0
    time_data = []
    state_data = []
    control_data = []
    
    glfw.init()
    window = glfw.create_window(1200, 900, "Optimized Pole Placement Segway", None, None)
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    while t_sim < duration:
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        u = -np.dot(K, (state - x_ref))
        data.ctrl[0] = u.item()
        mj.mj_step(model, data)
        
        time_data.append(t_sim)
        state_data.append(state.copy())
        control_data.append(u.item())
        
        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
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
    plt.plot(time_data, state_data[:, 0], 'b', label="Base Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Base Position (Optimized Pole Placement)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time_data, state_data[:, 2], 'r', label="Pendulum Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Pendulum Angle (Optimized Pole Placement)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time_data, control_data, 'g', label="Control Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Control (N)")
    plt.title("Control Signal (Optimized Pole Placement)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    best_params = optimize_controller()
    run_mujoco_simulation(best_params)

if __name__ == "__main__":
    main()
