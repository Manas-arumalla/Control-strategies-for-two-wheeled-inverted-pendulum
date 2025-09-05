import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Physical Parameters (from your MATLAB derivation)
# ----------------------------
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

# ----------------------------
# Sliding Mode Control (SMC) Parameters
# ----------------------------
lambda1 = 2.398643    # gain for cart position
lambda2 = 5.38826128   # gain for pendulum angle
lambda3 = 0.87279966    # gain for pendulum angular velocity
K_smc   = 18.6202205   # sliding mode control gain (tune for performance/chattering)
phi     = 0.0958816   # boundary layer thickness to reduce chattering

def sat(value):
    """Saturation function that limits its input to the range [-1, 1]."""
    return np.clip(value, -1.0, 1.0)

def compute_smc_control(x):
    """
    Compute the sliding mode control input.
    
    x: state vector [x1, x2, x3, x4] where:
       x1: cart position,
       x2: cart velocity,
       x3: pendulum angle,
       x4: pendulum angular velocity.
    """
    # Compute the sliding surface: s = lambda1*x1 + x2 + lambda2*x3 + lambda3*x4
    s = lambda1 * x[0] + x[1] + lambda2 * x[2] + lambda3 * x[3]
    
    # Compute the saturation value (using boundary layer phi)
    s_sat = sat(s / phi)
    
    # Compute the equivalent sliding surface derivative (excluding the control term)
    # s_dot_equiv = lambda1*x2 + lambda2*x4 + (d/Delta)*(-b + lambda3*a)*x3
    s_dot_equiv = lambda1 * x[1] + lambda2 * x[3] + (d / Delta) * (-b + lambda3 * a) * x[2]
    
    # Effective gain factor: k_u = (c - lambda3*b)/Delta
    k_u = (c - lambda3 * b) / Delta
    
    # Control law: u = (1/k_u) * (- s_dot_equiv - K_smc * s_sat)
    u = (1.0 / k_u) * (- s_dot_equiv - K_smc * s_sat)
    return u

# ----------------------------
# Load the MuJoCo segway model
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions (introduce a small disturbance)
data.qpos[0] = 0.0   # initial base (slider) displacement [m]
data.qpos[1] = 0.1   # initial pendulum (hinge) angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# ----------------------------
# Simulation Parameters
# ----------------------------
duration = 10.0               # Total simulation time (seconds)
dt = model.opt.timestep       # simulation timestep (e.g., 0.001 s)
sim_steps = int(duration / dt)

# Data storage for plotting later
time_data = []
state_data = []
control_data = []

# ----------------------------
# Initialize MuJoCo Visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Segway SMC Simulation", None, None)
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ----------------------------
# Simulation Loop
# ----------------------------
step_count = 0
t_sim = 0.0

while step_count < sim_steps:
    # Extract current state:
    # x[0]: base (slider) position, x[1]: base velocity,
    # x[2]: pendulum angle, x[3]: pendulum angular velocity.
    current_state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # Compute SMC control input
    u_current = compute_smc_control(current_state)
    
    # Apply control input (actuator "drive" applies force to the slider joint)
    data.ctrl[0] = u_current
    
    # Step the simulation forward
    mj.mj_step(model, data)
    
    # Record simulation data
    time_data.append(t_sim)
    state_data.append(current_state.copy())
    control_data.append(u_current)
    
    # Render the scene
    viewport = mj.MjrRect(0, 0, 1200, 900)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    t_sim += dt
    step_count += 1
    if glfw.window_should_close(window):
        break

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