import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import time

# ----------------------------
# Physical and Derived Parameters
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)          # moment of inertia of the pendulum [kg*m^2]
l = 0.4            # pendulum length [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters for state-space model
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# ----------------------------
# Construct State-Space Matrices (Continuous-time)
# State vector: x = [base position, base velocity, pendulum angle, pendulum angular velocity]^T
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

# Target state: balance at zero
x_ref = np.zeros((4,))

# ----------------------------
# Combined Controller Parameters
# ----------------------------
# Sliding Mode Controller (SMC) parameters
lam = 3.5       # sliding surface slope (using 'lam' because lambda is a reserved word)
eta = 5.0       # reaching gain
epsilon = 0.01  # smoothing factor for tanh

# Backstepping controller gains
k1, k2, k3, k4 = 8, 12, 6, 10

# Control weights for combining LQR, backstepping, and SMC
alpha = 0.7   # LQR influence weight
beta  = 0.2   # Backstepping weight
gamma = 0.1   # Sliding Mode weight

# Low-pass filter coefficient for smooth control input
alpha_filter = 0.05
filtered_u = 0.0

# ----------------------------
# Load the MuJoCo segway model
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions (a disturbance for the pendulum)
data.qpos[0] = 0.0   # base (slider) position [m]
data.qpos[1] = 0.2   # pendulum angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# ----------------------------
# Simulation Parameters
# ----------------------------
duration = 10.0            # simulation time in seconds
dt = model.opt.timestep    # simulation timestep (0.001 sec)
time_data = []
state_data = []
control_data = []

# ----------------------------
# Initialize MuJoCo Visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Segway Combined Control", None, None)
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
    # Extract current state from sensors:
    # data.qpos[0]: base position, data.qvel[0]: base velocity,
    # data.qpos[1]: pendulum angle, data.qvel[1]: pendulum angular velocity.
    state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # LQR control: u_lqr = -K*(state - x_ref)
    u_lqr = -np.dot(K, (state - x_ref))
    
    # Sliding Mode Control (SMC) with tanh smoothing
    # Here, x3: pendulum angle, x4: pendulum angular velocity.
    x3 = state[2]
    x4 = state[3]
    s = x3 + lam * x4
    u_smc = -eta * np.tanh(s / epsilon)
    
    # Backstepping Control
    x1 = state[0]
    x2 = state[1]
    v1 = -k1 * x3
    v2 = -k2 * x4
    v3 = -k3 * (x1 + x2)
    v4 = -k4 * (x3 + x4)
    u_bs = v1 + v2 + v3 + v4
    
    # Combined Control Strategy (weighted sum)
    u_raw = alpha * u_lqr + beta * u_bs + gamma * u_smc

    # Low-pass filter for smooth control input
    filtered_u = (1 - alpha_filter) * filtered_u + alpha_filter * u_raw
    u = filtered_u  # final control signal

    # Apply control input to the actuator (extract scalar if necessary)
    data.ctrl[0] = u.item() if isinstance(u, np.ndarray) else u

    # Step the simulation
    mj.mj_step(model, data)
    
    # Record simulation data
    time_data.append(t_sim)
    state_data.append(state.copy())
    control_data.append(u)
    
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
plt.title("Filtered Combined Control Signal")
plt.grid(True)

plt.tight_layout()
plt.show()
