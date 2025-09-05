import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# ----------------------------
# System Parameters
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)   # inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)           # inertia of the pendulum [kg*m^2]
l  = 0.4           # pendulum length (COM distance) [m]
r  = 0.0726        # wheel radius [m]
g  = 9.81         # gravitational acceleration [m/s^2]

# Derived parameters for dynamics and NMPC linearization
a = 2 * mw + mp + 2 * Iw/(r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# ----------------------------
# Linearized Model (for NMPC design)
# ----------------------------
# State vector: [base position; base velocity; pendulum angle; pendulum angular velocity]
A_lin = np.array([[0, 1,             0,            0],
                  [0, 0,    -b*d/Delta,            0],
                  [0, 0,             0,            1],
                  [0, 0,     a*d/Delta,           0]])
B_lin = np.array([[0],
                  [c/Delta],
                  [0],
                  [-b/Delta]])

# ----------------------------
# Controller Tuning Parameters
# ----------------------------

# NMPC parameters
N_horizon = 10             # prediction horizon (steps)
Q_nmpc = np.diag([10, 1, 100, 1])   # state weighting
R_nmpc = 0.01              # control weighting (scalar)
u_sat = 10.0               # saturation limit on control input

# DSC (Dynamic Surface Control) parameters
lambda_dsc = 3.0           # DSC filter coefficient
alpha_dsc  = 20.0          # DSC gain
z = 0.0                    # DSC filter state (initialize to zero)

# Adaptive NN parameters (using a scalar weight)
nn_w = 0.0                 # initial NN weight
gamma_nn = 0.05            # learning rate for NN weight update
nn_w_max = 5.0             # clamping limits
nn_w_min = -5.0

# ----------------------------
# NMPC Cost Function Definition
# ----------------------------
def nmpc_cost(u_seq, x0, A, B, Q, R, N_horizon):
    """
    Compute the quadratic cost over the prediction horizon.
    
    Parameters:
        u_seq: sequence of control inputs (array of length N_horizon)
        x0: current state (4-vector)
        A, B: linearized state-space matrices
        Q: state cost weighting (4x4 matrix)
        R: control input weighting (scalar)
        N_horizon: prediction horizon (integer)
        
    Returns:
        J: cumulative cost
    """
    # Ensure u_seq is a 1D array of length N_horizon
    u_seq = np.reshape(u_seq, (N_horizon,))
    x_pred = np.copy(x0)
    J = 0.0
    for k in range(N_horizon):
        u = u_seq[k]
        # Euler integration with linear model
        x_pred = A.dot(x_pred) + B.flatten() * u
        J += x_pred.T.dot(Q).dot(x_pred) + R*(u**2)
    if not np.isfinite(J):
        J = 1e6
    return J

# ----------------------------
# MuJoCo Simulation Setup
# ----------------------------
# Load the MuJoCo segway model (ensure "segway.xml" exists in your working directory)
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions (apply a disturbance to the pendulum angle)
data.qpos[0] = 0.0    # base position [m]
data.qpos[1] = 0.2    # pendulum angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# ----------------------------
# Simulation Parameters
# ----------------------------
duration = 10.0                 # simulation duration in seconds
dt = model.opt.timestep         # simulation timestep (e.g., 0.001 sec)
time_data = []
state_data = []
control_data = []

# ----------------------------
# Initialize MuJoCo Visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Segway Hybrid Control: DSC+NN+NMPC", None, None)
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ----------------------------
# Main Simulation Loop
# ----------------------------
t_sim = 0.0
while t_sim < duration:
    # ----------------------------
    # State Extraction
    # ----------------------------
    # For this model, we assume:
    #   data.qpos[0]: base position, data.qvel[0]: base velocity,
    #   data.qpos[1]: pendulum angle, data.qvel[1]: pendulum angular velocity.
    state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # ----------------------------
    # Outer-Loop NMPC
    # ----------------------------
    # Set up the NMPC optimization problem using the linearized model.
    # Use the current state as the initial condition.
    u0 = np.zeros(N_horizon)  # initial guess for the control sequence
    bounds = [(-u_sat, u_sat)] * N_horizon  # control bounds for each step
    
    sol = minimize(nmpc_cost, u0, args=(state, A_lin, B_lin, Q_nmpc, R_nmpc, N_horizon),
                   bounds=bounds, method='SLSQP', options={'disp': False})
    
    # Use the first control action from the optimized sequence
    u_nmpc = sol.x[0]
    
    # ----------------------------
    # Inner-Loop DSC (Dynamic Surface Control)
    # ----------------------------
    # Define the error as the deviation of the pendulum angle from zero.
    e = state[2]         # pendulum angle error (desired angle = 0)
    e_dot = state[3]     # angular velocity of the pendulum
    # Update the DSC filter state using Euler integration:
    z = (1 - dt * lambda_dsc) * z + dt * e_dot
    u_dsc = -alpha_dsc * (e + z)
    
    # ----------------------------
    # Adaptive NN Augmentation
    # ----------------------------
    u_nn = nn_w * e
    # Update the NN weight using gradient descent on squared error
    nn_w = nn_w + gamma_nn * (e**2)
    # Clamp the NN weight to prevent windup
    nn_w = np.clip(nn_w, nn_w_min, nn_w_max)
    
    # ----------------------------
    # Combine Control Inputs and Apply Saturation
    # ----------------------------
    u_combined = u_nmpc + u_dsc + u_nn
    u_total = np.clip(u_combined, -u_sat, u_sat)
    
    # ----------------------------
    # Apply Control Input to the MuJoCo Model
    # ----------------------------
    # Assume actuator index 0 controls the torque/force for the base.
    data.ctrl[0] = u_total
    
    # ----------------------------
    # Step the Simulation
    # ----------------------------
    mj.mj_step(model, data)
    
    # Record simulation data
    time_data.append(t_sim)
    state_data.append(state.copy())
    control_data.append(u_total)
    
    # ----------------------------
    # Render the Scene
    # ----------------------------
    viewport = mj.MjrRect(0, 0, 1200, 900)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    t_sim += dt
    if glfw.window_should_close(window):
        break

glfw.terminate()

# ----------------------------
# Plot Simulation Results
# ----------------------------
time_data = np.array(time_data)
state_data = np.array(state_data)
control_data = np.array(control_data)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time_data, state_data[:, 0], 'b', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Base Position (m)")
plt.title("Hybrid Controller (DSC+NN+NMPC): Base Position")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_data, state_data[:, 2], 'r', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Pendulum Angle (rad)")
plt.title("Hybrid Controller (DSC+NN+NMPC): Pendulum Angle")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_data, control_data, 'k', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Control Input (N)")
plt.title("Hybrid Controller (DSC+NN+NMPC): Combined Control Effort")
plt.grid(True)

plt.tight_layout()
plt.show()
