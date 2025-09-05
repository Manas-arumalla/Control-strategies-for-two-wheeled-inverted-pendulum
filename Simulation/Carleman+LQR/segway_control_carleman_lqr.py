import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import control as ctrl  
import matplotlib.pyplot as plt

# ----------------------------
# 1. Define Physical Parameters
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
r = 0.0726         # wheel radius [m]
l = 0.4            # pendulum length [m]
g = 9.81           # gravitational acceleration [m/s^2]
Iw = 0.5 * mw * (r**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (l**2)       # moment of inertia of the pendulum [kg*m^2]

# ----------------------------
# 2. Derived Parameters and Augmented Model Setup
# ----------------------------
a = 2 * mw + mp + 2 * Iw/(r**2)
b = mp * l
c = mp * (l**2) + Ip
denom = a * c - b**2
d = mp * g * l  # shorthand for mp*g*l

# Tuning parameters for Carleman augmentation
alpha_val = 1.0    # tuning parameter for auxiliary dynamics
delta_val = 0.05   # coupling from control input into auxiliary state

# Augmented state-space matrices (5 states):
# Original states: x, x_dot, theta, theta_dot;
# Auxiliary state: z5 with dynamics: z5_dot = theta_dot + alpha*(theta - z5) + delta*u.
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

# ----------------------------
# 3. LQR Controller Design for the Augmented Model
# ----------------------------
# Increase weight on the cart position (x) and set moderate penalty for z5.
Q_aug = np.diag([100, 1, 100, 1, 10])
R = np.array([[1]])
K, S, E = ctrl.lqr(A_aug, B_aug, Q_aug, R)
K = np.asarray(K)
print("Augmented LQR Gain K:", K)

# ----------------------------
# 4. Load MuJoCo Model and Set Initial Conditions
# ----------------------------
# Ensure that "segway.xml" is in your working directory.
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial physical states:
data.qpos[0] = 0.0   # Base (slider) position (x)
data.qpos[1] = 0.2   # Pendulum angle (theta) in radians
data.qvel[0] = 0.0   # Base velocity (x_dot)
data.qvel[1] = 0.0   # Pendulum angular velocity (theta_dot)

# Initialize the auxiliary state z5 (set equal to the initial pendulum angle)
z5 = data.qpos[1].item()

# ----------------------------
# 5. Simulation Setup
# ----------------------------
duration = 10.0          # simulation duration in seconds
dt = model.opt.timestep  # simulation timestep (e.g., 0.001 sec)

# Data recording lists
time_data = []
state_data = []      # physical states: [x, x_dot, theta, theta_dot]
z5_data = []         # auxiliary state data
control_data = []    # control input history

# ----------------------------
# 6. Initialize MuJoCo Visualization (GLFW)
# ----------------------------
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1200, 900, "Carleman Linearization with LQR", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ----------------------------
# 7. Simulation Loop
# ----------------------------
t_sim = 0.0
while t_sim < duration:
    # Extract current physical states and ensure they are scalars using .item()
    x = data.qpos[0].item()
    theta = data.qpos[1].item()
    x_dot = data.qvel[0].item()
    theta_dot = data.qvel[1].item()
    
    # Build the augmented state vector: [x, x_dot, theta, theta_dot, z5]
    z = np.array([x, x_dot, theta, theta_dot, z5])
    
    # Compute LQR control and force it to be a scalar using .item()
    u = -np.dot(K, z).item()
    
    # Apply control input to the actuator.
    data.ctrl[0] = u
    
    # Update the auxiliary state z5 using Euler integration:
    # z5_dot = theta_dot + alpha*(theta - z5) + delta*u.
    z5_dot = theta_dot + alpha_val * (theta - z5) + delta_val * u
    z5 = z5 + dt * z5_dot

    # Step the simulation.
    mj.mj_step(model, data)
    
    # Record simulation data.
    time_data.append(t_sim)
    state_data.append([x, x_dot, theta, theta_dot])
    z5_data.append(z5)
    control_data.append(u)
    
    # Render the scene.
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
# 8. Plot the Simulation Results
# ----------------------------
time_data = np.array(time_data)
state_data = np.array(state_data)
z5_data = np.array(z5_data)
control_data = np.array(control_data)

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_data, state_data[:, 0], 'b', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("x (m)")
plt.title("Base (Slider) Position")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time_data, state_data[:, 2], 'r', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("theta (rad)")
plt.title("Pendulum Angle")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time_data, z5_data, 'm', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("z5")
plt.title("Auxiliary State z5")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time_data, control_data, 'g', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Control Input (N)")
plt.title("LQR Control Input")
plt.grid(True)

plt.tight_layout()
plt.show()
