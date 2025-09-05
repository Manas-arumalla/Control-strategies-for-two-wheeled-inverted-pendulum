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

# Derived parameters from the linearized model
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# Construct the state-space matrices (continuous-time)
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
# Pole Placement Controller Design
# ----------------------------
# Desired closed-loop poles (you can modify these to tune performance)
desired_poles = [-43.9360, -0.7630, -2.0485, -48.7724]

# Compute the state-feedback gain using pole placement:
K = ctrl.place(A, B, desired_poles)
K = np.asarray(K)
print("Pole Placement Gain K:", K)

# Optionally, verify the closed-loop eigenvalues:
A_cl = A - np.dot(B, K)
eig_vals = np.linalg.eigvals(A_cl)
print("Closed-loop eigenvalues:", eig_vals)

# Target state: we want the system to balance at zero
x_ref = np.zeros((4,))

# ----------------------------
# Load the MuJoCo segway model
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set an initial disturbance for the controller to correct
data.qpos[0] = 0.0   # initial slider (base) displacement [m]
data.qpos[1] = 0.1   # initial pendulum angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# ----------------------------
# Simulation Parameters
# ----------------------------
duration = 10.0      # simulation duration in seconds
dt = model.opt.timestep  # simulation timestep from the model (e.g., 0.001 sec)
time_data = []
state_data = []
control_data = []

# ----------------------------
# Initialize MuJoCo Visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Self-Balancing Robot (Pole Placement)", None, None)
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
    # Extract current state:
    # data.qpos[0]: slider (base) position (x)
    # data.qpos[1]: pendulum (hinge) angle (theta)
    # data.qvel[0]: slider velocity (x_dot)
    # data.qvel[1]: pendulum angular velocity (theta_dot)
    state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # Compute control: u = -K * (state - x_ref)
    u = -np.dot(K, (state - x_ref))
    
    # Apply the control input to the actuator
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
plt.plot(time_data, state_data[:, 0], 'b', label="Base Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Base Position (Pole Placement)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_data, state_data[:, 2], 'r', label="Pendulum Angle")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Pendulum Angle (Pole Placement)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_data, control_data, 'g', label="Control Input")
plt.xlabel("Time (s)")
plt.ylabel("Control (N)")
plt.title("Control Signal (Pole Placement)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
