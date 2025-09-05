import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import cont2discrete

# ----------------------------
# Plant Parameters (from MATLAB derivation)
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)   # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)           # moment of inertia of the pendulum [kg*m^2]
l = 0.4            # length of the pendulum [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters for the linearized model
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * l**2 + Ip
d = mp * g * l
Delta = a * c - b**2

# Continuous-time state-space matrices (states: [cart pos; cart vel; pendulum angle; pendulum angular vel])
A_cont = np.array([[0, 1,          0,      0],
                   [0, 0,  -b*d/Delta,      0],
                   [0, 0,          0,      1],
                   [0, 0,   a*d/Delta,      0]])
B_cont = np.array([[0],
                   [c/Delta],
                   [0],
                   [-b/Delta]])
# Output matrix (cart position and pendulum angle)
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# ----------------------------
# Discretize the Model for Controller Design
# ----------------------------
Ts = 0.1  # EPSAC sampling time (seconds)
# Use scipy.signal.cont2discrete to get the discrete-time model.
sysd = cont2discrete((A_cont, B_cont, C, np.zeros((2,1))), Ts)
A_d = sysd[0]
B_d = sysd[1]

# ----------------------------
# EPSAC Design Parameters
# ----------------------------
Np = 20            # Prediction Horizon (number of steps)
Nc = 5             # Control Horizon (number of optimized control moves)
lam = 0.1          # Weight on control effort in the cost function
u_min = -10        # Minimum control input
u_max = 10         # Maximum control input

# Initialize the estimated model as the discretized (nominal) model.
A_est = A_d.copy()
B_est = B_d.copy()

# ----------------------------
# MuJoCo Simulation Setup
# ----------------------------
# Load the segway model (ensure segway.xml is in your working directory)
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions (introduce a small disturbance)
data.qpos[0] = 0.0   # initial base (slider) displacement [m]
data.qpos[1] = 0.1   # initial pendulum (hinge) angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# Get the MuJoCo simulation timestep (may be different from Ts)
sim_dt = model.opt.timestep  # e.g., 0.001 s
duration = 10.0              # Total simulation time (seconds)
sim_steps = int(duration / sim_dt)

# Define the control update interval (in simulation steps)
control_interval = int(Ts / sim_dt)  # update EPSAC every Ts seconds

# Data storage for plotting later
time_data = []
state_data = []
control_data = []

# ----------------------------
# Define the EPSAC Cost Function
# ----------------------------
def epsac_cost(U, x0, A_est, B_est, Nc, Np, C, lam):
    """
    Compute the cost over the prediction horizon for a candidate control sequence.
    U: vector of control moves of length Nc.
    x0: current state (4x1).
    The control moves are extended by holding the last value constant.
    """
    # Extend U to length Np by repeating the last element
    U_extended = np.concatenate([U, np.full(Np - Nc, U[-1])])
    x_pred = x0.copy()
    cost = 0.0
    # For this simulation, the reference is zero for both outputs.
    ref = np.zeros((C.shape[0],))
    for i in range(Np):
        # Predict next state using the estimated model
        x_pred = A_est @ x_pred + B_est.flatten() * U_extended[i]
        y_pred = C @ x_pred
        e = ref - y_pred
        cost += np.dot(e, e) + lam * (U_extended[i]**2)
    return cost

# ----------------------------
# Main Simulation and EPSAC Loop
# ----------------------------
# Initialize GLFW for visualization
glfw.init()
window = glfw.create_window(1200, 900, "Segway EPSAC Simulation", None, None)
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# For EPSAC updates, we will extract the state as:
# x = [qpos[0], qvel[0], qpos[1], qvel[1]]
def get_state(data):
    return np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])

# EPSAC control update parameters
adaptation_gain = 0.001

# Initialize current control input
u_current = 0.0

# Simulation loop counters
step_count = 0
t_sim = 0.0

while step_count < sim_steps:
    # At control update instants, compute the EPSAC control action.
    if step_count % control_interval == 0:
        # Get current state for controller update
        xk = get_state(data)
        
        # Initial guess for optimization (Nc-length vector)
        U0 = np.zeros(Nc)
        # Set bounds for each control input
        bounds = [(u_min, u_max)] * Nc
        
        # Optimize using SLSQP (similar to MATLAB fmincon)
        res = minimize(epsac_cost, U0, args=(xk, A_est, B_est, Nc, Np, C, lam),
                       method='SLSQP', bounds=bounds, options={'disp': False})
        
        if res.success:
            U_opt = res.x
        else:
            print("Optimization failed at t = {:.3f}s".format(t_sim))
            U_opt = U0
        
        # Apply only the first control move from the optimized sequence.
        u_current = U_opt[0]
        
        # (Adaptive update of the estimated model)
        # Predict the next state using the estimated model:
        pred_x = A_est @ xk + B_est.flatten() * u_current
        # (After one control interval, we will compare with measured state.)
    
    # Apply the computed control input to the simulation.
    # Assume the actuator "drive" applies force to the base joint.
    data.ctrl[0] = u_current
    
    # Step the simulation forward by one simulation timestep.
    mj.mj_step(model, data)
    
    # At the end of a control interval, update the estimated model.
    if (step_count + 1) % control_interval == 0:
        x_next = get_state(data)
        error_x = x_next - pred_x
        # Update A_est and B_est using outer products.
        A_est = A_est + adaptation_gain * np.outer(error_x, xk)
        B_est = B_est + adaptation_gain * np.outer(error_x, np.array([u_current])).reshape(B_est.shape)
    
    # Record data for plotting at control update instants
    if step_count % control_interval == 0:
        time_data.append(t_sim)
        state_data.append(get_state(data).copy())
        control_data.append(u_current)
    
    # Render the scene every simulation step.
    viewport = mj.MjrRect(0, 0, 1200, 900)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # Check if window was closed.
    if glfw.window_should_close(window):
        break
    
    t_sim += sim_dt
    step_count += 1

glfw.terminate()

# ----------------------------
# Plot Simulation Results
# ----------------------------
time_data = np.array(time_data)
state_data = np.array(state_data)
control_data = np.array(control_data)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_data, state_data[:, 0], 'b', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Base Position (m)")
plt.title("Base (Slider) Position")
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
plt.title("Control Signal")
plt.grid(True)

plt.tight_layout()
plt.show()
