import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import control as ctrl  
import matplotlib.pyplot as plt
import time  

# Define physical parameters 
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)          # moment of inertia of the pendulum [kg*m^2]
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
# Construct state-space matrices (continuous-time)
# State vector: x = [ x, x_dot, theta, theta_dot ]^T
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

# ----------------------------
# LQR Controller Design
# ----------------------------
Q = np.diag([100, 1, 1000, 1])
R = np.array([[0.01]])
K, S, E = ctrl.lqr(A, B, Q, R)
K = np.asarray(K)
print("LQR Gain K:", K)

# Target state (we want the system to balance at zero)
x_ref = np.zeros((4,))

# ----------------------------
# Load the MuJoCo segway model
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set an initial disturbance so the controller has an error to correct
data.qpos[0] = 0.0   # initial slider (horizontal) displacement (in meters)
data.qpos[1] = 0.2   # initial pendulum angle (in radians)
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# ----------------------------
# Simulation parameters
# ----------------------------
duration = 10.0      # seconds
dt = model.opt.timestep  # simulation timestep (e.g., 0.001 sec)
time_data = []
state_data = []
control_data = []

# ----------------------------
# Initialize MuJoCo visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Self-Balancing Robot (LQR)", None, None)
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ----------------------------
# Simulation Loop (normal speed)
# ----------------------------
t_sim = 0.0
while t_sim < duration:
    # Extract current state:
    # data.qpos[0]: slider (base) position (x)
    # data.qpos[1]: pendulum (hinge) angle (theta)
    # data.qvel[0]: slider velocity (x_dot)
    # data.qvel[1]: pendulum angular velocity (theta_dot)
    state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # Compute LQR control: u = -K*(state - x_ref)
    u = -np.dot(K, (state - x_ref))
    
    # Apply control input to the actuator (using .item() to extract scalar)
    data.ctrl[0] = u.item()
    
    # Step the simulation
    mj.mj_step(model, data)
    
    # Record simulation data
    time_data.append(t_sim)
    state_data.append(state.copy())
    control_data.append(u.item())
    
    # Render the scene
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
# Plot the Simulation Results
# ----------------------------
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
plt.title("Control Signal")
plt.grid(True)

plt.tight_layout()
plt.show()