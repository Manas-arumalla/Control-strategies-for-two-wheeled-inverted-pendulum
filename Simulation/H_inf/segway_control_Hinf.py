import numpy as np
import control as ctrl      # pip install control (with slycot for hinfsyn)
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt

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
Wp = (s + 10) / (s + 0.1)   # performance weight
Wu = ctrl.tf(0.1, 1)        # control weight (now a transfer function)

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
# 3. MuJoCo Simulation Setup with H∞ Controller
# =============================================================================
# Load your MuJoCo model (ensure "segway.xml" is in your working directory)
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions (small initial angle to remain in the linear regime)
data.qpos[0] = 0.0    # cart (slider) position [m]
data.qpos[1] = 0.1    # pendulum angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# Simulation parameters
duration = 10.0           # simulation time (seconds)
dt = model.opt.timestep   # simulation timestep (e.g., 0.001 s)
t_sim = 0.0

# Lists for data recording
time_data = []
state_data = []   # [cart pos, cart vel, pend angle, pend ang vel]
control_data = []

# =============================================================================
# 4. Initialize MuJoCo Visualization (GLFW)
# =============================================================================
if not glfw.init():
    raise Exception("GLFW initialization failed")
window = glfw.create_window(1200, 900, "H∞ Controlled Segway", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# =============================================================================
# 5. Simulation Loop
# =============================================================================
while t_sim < duration:
    # Measured outputs from simulation:
    # data.qpos[0] is the cart position; data.qpos[1] is the pendulum angle.
    y = np.array([data.qpos[0], data.qpos[1]])
    
    # Update the H∞ controller dynamics using Euler integration
    xK_dot = A_hinf.dot(xK) + B_hinf.dot(y)
    xK = xK + dt * xK_dot
    
    # Compute control input: u = C_hinf*xK + D_hinf*y
    u = C_hinf.dot(xK) + D_hinf.dot(y)
    
    # Apply control input to the slider joint actuator
    data.ctrl[0] = u.item()
    
    # Step the simulation forward
    mj.mj_step(model, data)
    
    # Record simulation data
    time_data.append(t_sim)
    state_data.append([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
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

# =============================================================================
# 6. Plot Simulation Results
# =============================================================================
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
