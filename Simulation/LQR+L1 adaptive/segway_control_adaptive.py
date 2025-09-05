import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_continuous_lyapunov

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
# State-Space Model (Continuous-Time)
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
# LQR Controller Design
# ----------------------------
# Use a Q weighting to emphasize cart position stabilization, as in the MATLAB code
Q = np.diag([1000, 1, 100, 1])
R = np.array([[0.01]])
K, _, _ = ctrl.lqr(A, B, Q, R)
K = np.asarray(K)
print("LQR Gain K:", K)

# Compute closed-loop A matrix and Lyapunov matrix P
A_cl = A - B @ K
# Solve A_cl.T*P + P*A_cl = -I (4x4 identity)
P = solve_continuous_lyapunov(A_cl.T, -np.eye(4))

# ----------------------------
# L1 Adaptive Controller Parameters
# ----------------------------
Gamma = 50       # adaptation gain
tau = 0.05       # low-pass filter time constant
sigma_max = 10   # projection bound for the adaptive estimate

# ----------------------------
# MuJoCo Model and Initialization
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set initial conditions:
# Here we set the base (slider) to 0 and the pendulum angle to 0.1 rad.
data.qpos[0] = 0.0   # base (slider) position [m]
data.qpos[1] = 0.1   # pendulum angle [rad]
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# Identify the pendulum body index for applying disturbance torque later
pendulum_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pendulum")

# ----------------------------
# Initialize Adaptive Controller States
# ----------------------------
x_hat = np.zeros((4,))    # predictor state
sigma_hat = 0.0           # adaptive estimate (scalar)
z_f = 0.0                 # low-pass filter state

# ----------------------------
# Simulation Parameters
# ----------------------------
duration = 10.0      # seconds
dt = model.opt.timestep  # simulation timestep (e.g., 0.001 sec)

# Data logging arrays
time_data = []
state_data = []      # plant state: [x, x_dot, theta, theta_dot]
control_data = []    # control input
sigma_data = []      # adaptive estimate
zf_data = []         # filtered term

# ----------------------------
# Initialize MuJoCo Visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Self-Balancing Robot (L1 Adaptive Control)", None, None)
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
    # Get current plant state from MuJoCo
    # data.qpos[0]: slider (base) position
    # data.qpos[1]: pendulum angle
    # data.qvel[0]: slider velocity
    # data.qvel[1]: pendulum angular velocity
    x = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    
    # Compute control input using L1 adaptive law:
    # u = -K*x - z_f
    u = - np.dot(K, x) - z_f

    # (Optional) External disturbance on the pendulum dynamics:
    # Compute a disturbance torque (acts along the pendulum hinge, y-axis)
    d_ext = 0.5 * np.sin(0.5 * t_sim)
    # Apply the disturbance torque to the pendulum body.
    # The spatial force vector: [fx, fy, fz, tx, ty, tz] where we apply torque about y.
    data.xfrc_applied[pendulum_body_id, :] = np.array([0, 0, 0, 0, d_ext, 0])
    
    # Apply control input to the slider joint actuator.
    data.ctrl[0] = u.item()  # ensure scalar value
    
    # Step the MuJoCo simulation
    mj.mj_step(model, data)
    
    # ----------------------------
    # Update the Adaptive Controller Dynamics (Euler integration)
    # ----------------------------
    # Predictor dynamics: x_hat_dot = A*x_hat + B*(u + sigma_hat)
    x_hat_dot = A @ x_hat + (B.flatten() * (u + sigma_hat))
    x_hat = x_hat + dt * x_hat_dot
    
    # Adaptation law (with projection):
    # sigma_dot_temp = Gamma * B' * P * (x - x_hat)
    sigma_dot_temp = Gamma * (B.T @ P @ (x - x_hat)).item()
    if (sigma_hat >= sigma_max and sigma_dot_temp > 0) or (sigma_hat <= -sigma_max and sigma_dot_temp < 0):
        sigma_dot = 0.0
    else:
        sigma_dot = sigma_dot_temp
    sigma_hat = sigma_hat + dt * sigma_dot
    
    # Low-pass filter dynamics: z_f_dot = (-1/tau)*z_f + (1/tau)*sigma_hat
    z_f_dot = (-1.0/tau)*z_f + (1.0/tau)*sigma_hat
    z_f = z_f + dt * z_f_dot

    # ----------------------------
    # Logging Data
    # ----------------------------
    time_data.append(t_sim)
    state_data.append(x.copy())
    control_data.append(u.item())
    sigma_data.append(sigma_hat)
    zf_data.append(z_f)
    
    # ----------------------------
    # Render the Scene
    # ----------------------------
    viewport = mj.MjrRect(0, 0, 1200, 900)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # Advance simulation time
    t_sim += dt

    if glfw.window_should_close(window):
        break

# Terminate GLFW
glfw.terminate()

# ----------------------------
# Plot Simulation Results
# ----------------------------
time_data = np.array(time_data)
state_data = np.array(state_data)
control_data = np.array(control_data)
sigma_data = np.array(sigma_data)
zf_data = np.array(zf_data)

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_data, state_data[:, 0], label="Cart Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Cart Position")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time_data, state_data[:, 2], label="Pendulum Angle", color="r")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Pendulum Angle")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time_data, control_data, label="Control Input", color="g")
plt.xlabel("Time (s)")
plt.ylabel("Control (N)")
plt.title("Control Effort")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time_data, sigma_data, 'm', label="Adaptive Estimate")
plt.plot(time_data, zf_data, 'c--', label="Filtered Term")
plt.xlabel("Time (s)")
plt.ylabel("Adaptive Terms")
plt.title("Adaptive Estimate (σ̂) and Filtered Term (z_f)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
