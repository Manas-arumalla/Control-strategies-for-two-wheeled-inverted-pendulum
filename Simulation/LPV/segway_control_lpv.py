import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

# ----------------------------
# System Parameters
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)   # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)           # moment of inertia of the pendulum [kg*m^2]
l = 0.4            # length of the pendulum [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# ----------------------------
# State-Space Model (Nominal)
# ----------------------------
# State vector: x = [ cart position; cart velocity; pendulum angle; pendulum angular velocity ]
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

# ----------------------------
# LPV Gain-Scheduling Design
# ----------------------------
# Define the range for the scheduling parameter (pendulum angle deviation, in rad)
p_range = np.linspace(-0.1, 0.1, 5)  # e.g., from -0.1 to 0.1 rad
num_points = len(p_range)

# LQR weighting matrices (similar to your MATLAB code)
Q = np.diag([100, 1, 1000, 1])
R = np.array([[0.01]])

# Preallocate array to store LQR gains (each gain is a row vector of 4 elements)
K_schedule = np.zeros((num_points, 4))

for i in range(num_points):
    p = p_range[i]
    # Modify gravitational effect for the pendulum (small angle approximation)
    d_eff = mp * g * l * np.cos(p)
    
    # Create a modified copy of A: only the (4,3) element (row index 3, col index 2) changes.
    A_lpv = A.copy()
    A_lpv[3, 2] = a * d_eff / Delta
    
    # Compute LQR gain at this operating point.
    # ctrl.lqr returns (K, S, E); we only need K.
    K, _, _ = ctrl.lqr(A_lpv, B, Q, R)
    # Ensure K is a 1-D array
    K_schedule[i, :] = np.asarray(K).flatten()

# Create an interpolation function for the gain
# This function maps any pendulum angle p_val to a gain vector (4,)
K_interp_func = interp1d(p_range, K_schedule, kind='linear', fill_value="extrapolate", axis=0)

# ----------------------------
# MuJoCo Model and Initialization
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions:
# data.qpos[0]: cart (slider) position, data.qpos[1]: pendulum angle
data.qpos[0] = 0.0   # cart position [m]
data.qpos[1] = 0.1   # pendulum angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# Identify the pendulum body index (used if you want to apply any disturbance)
pendulum_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pendulum")

# ----------------------------
# Simulation Parameters and Logging
# ----------------------------
duration = 10.0      # total simulation time (seconds)
dt = model.opt.timestep  # simulation timestep from model options

# Arrays to log simulation data
time_data = []
state_data = []      # plant state: [cart pos, cart vel, pendulum angle, pendulum angular vel]
control_data = []    # control input

# ----------------------------
# Initialize MuJoCo Visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "LPV Controller in MuJoCo", None, None)
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ----------------------------
# Simulation Loop
# ----------------------------
t_sim = 0.0
while t_sim < duration:
    # Get current plant state from MuJoCo.
    # x = [cart position, cart velocity, pendulum angle, pendulum angular velocity]
    x = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # Scheduling parameter: use current pendulum angle (x[2])
    p_current = x[2]
    
    # Interpolate to get current gain
    K_current = K_interp_func(p_current)  # returns a (4,) vector
    
    # Compute control input: u = -K(x)*x
    u = -np.dot(K_current, x)
    
    # (Optional) Apply an external disturbance torque on the pendulum (if desired)
    d_ext = 0.5 * np.sin(0.5 * t_sim)
    # The spatial force vector format: [fx, fy, fz, tx, ty, tz]
    data.xfrc_applied[pendulum_body_id, :] = np.array([0, 0, 0, 0, d_ext, 0])
    
    # Apply the control input to the cart (slider joint actuator).
    data.ctrl[0] = u.item()  # ensure scalar value
    
    # Step the MuJoCo simulation
    mj.mj_step(model, data)
    
    # Log data
    time_data.append(t_sim)
    state_data.append(x.copy())
    control_data.append(u.item())
    
    # Render the scene
    viewport = mj.MjrRect(0, 0, 1200, 900)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # Advance simulation time
    t_sim += dt
    
    if glfw.window_should_close(window):
        break

# Terminate GLFW after simulation
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
plt.title("Control Effort")
plt.grid(True)

plt.tight_layout()
plt.show()
