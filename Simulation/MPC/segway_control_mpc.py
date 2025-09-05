import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import time

# ----------------------------
# Define physical parameters (from your MATLAB derivation)
# ----------------------------
mw = 0.432         # mass of each wheel [kg]
mp = 5.0           # mass of the pendulum [kg]
Iw = 0.5 * mw * (0.0726**2)  # moment of inertia of each wheel [kg*m^2]
Ip = 5.0 * (0.4**2)          # moment of inertia of the pendulum [kg*m^2]
l = 0.4            # length of the pendulum [m]
r = 0.0726         # wheel radius [m]
g = 9.81           # gravitational acceleration [m/s^2]

# Derived parameters from the linearized model:
a = 2 * mw + mp + 2 * Iw / (r**2)
b = mp * l
c = mp * (l**2) + Ip
d = mp * g * l
Delta = a * c - b**2

# ----------------------------
# Continuous-time state-space matrices
# States: x = [cart position, cart velocity, pendulum angle, pendulum angular velocity]^T
# ----------------------------
A = np.array([[0,      1,           0,           0],
              [0,      0,   -b*d/Delta,           0],
              [0,      0,           0,           1],
              [0,      0,    a*d/Delta,          0]])
B = np.array([[0],
              [c/Delta],
              [0],
              [-b/Delta]])

# Output matrix: we are interested in [cart position; pendulum angle]
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# ----------------------------
# Discretize the system for MPC using a sampling time Ts = 0.1 s
# ----------------------------
Ts = 0.05  # MPC sampling time (seconds)
sysd = cont2discrete((A, B, np.eye(4), np.zeros((4,1))), Ts)
A_d = sysd[0]
B_d = sysd[1]

# ----------------------------
# MPC Parameters and Weights
# ----------------------------
N = 20          # Prediction horizon (steps)
Nu = 5          # Control horizon (steps) after which control is held constant

# Output tracking weights: emphasize pendulum angle error over cart position error.
W_out = np.diag([100, 1000])
R_weight = 0.01            # Weight on control input magnitude
Delta_u_weight = 0.1       # Weight on change in control input (rate penalty)

# Terminal cost: compute P using the discrete algebraic Riccati equation.
Q_state = np.diag([100, 1, 1000, 1])
R_state = np.array([[0.01]])
P_terminal = solve_discrete_are(A_d, B_d, Q_state, R_state)

# Control input constraints:
u_min = -10
u_max = 10

# ----------------------------
# MPC Solver using CVXPY with fallback solver
# ----------------------------
def solve_mpc(x0, u_prev, ref, A_d, B_d, N, Nu, P_terminal):
    n = A_d.shape[0]  # number of states (4)
    m = B_d.shape[1]  # number of inputs (1)
    
    # Decision variables for state trajectory and control inputs:
    x = cp.Variable((n, N+1))
    u = cp.Variable((m, N))
    
    cost = 0
    constraints = []
    
    # Initial condition constraint:
    constraints += [x[:, 0] == x0]
    
    for k in range(N):
        # Dynamics constraint: x[k+1] = A_d * x[k] + B_d @ u[k]
        constraints += [x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k]]
        
        # Input constraints:
        constraints += [u[:, k] >= u_min, u[:, k] <= u_max]
        
        # Enforce constant control beyond the control horizon:
        if k >= Nu:
            constraints += [u[:, k] == u[:, Nu-1]]
        
        # Output tracking cost (tracking cart position and pendulum angle):
        y_k = C @ x[:, k]
        cost += cp.quad_form(y_k - ref, W_out)
        
        # Cost on control input magnitude:
        cost += R_weight * cp.sum_squares(u[:, k])
        
        # Cost on the rate of change of control input:
        if k == 0:
            cost += Delta_u_weight * cp.sum_squares(u[:, k] - u_prev)
        else:
            cost += Delta_u_weight * cp.sum_squares(u[:, k] - u[:, k-1])
    
    # Terminal cost:
    cost += cp.quad_form(x[:, N], P_terminal)
    
    # Define and solve the optimization problem:
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=True,
                   max_iter=10000, eps_abs=1e-4, eps_rel=1e-4)
    except cp.SolverError:
        print("OSQP failed, switching to ECOS...")
        prob.solve(solver=cp.ECOS, verbose=True)
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return u[:, 0].value.item()
    else:
        print("MPC optimization failed. Status:", prob.status)
        return 0.0

# ----------------------------
# Load the MuJoCo segway model
# ----------------------------
model = mj.MjModel.from_xml_path("segway.xml")
data = mj.MjData(model)

# Set an initial disturbance for the simulation:
data.qpos[0] = 0.0   # initial base (slider) displacement (m)
data.qpos[1] = 0.1   # initial pendulum (hinge) angle (rad)
data.qvel[0] = 0.0
data.qvel[1] = 0.0

# ----------------------------
# Simulation parameters
# ----------------------------
duration = 10.0      # Total simulation time (seconds)
dt = model.opt.timestep  # simulation time step (e.g., 0.001 s)
sim_steps = int(duration / dt)

# For MPC, update every Ts seconds (i.e. every Ts/dt simulation steps)
mpc_update_steps = int(Ts / dt)
u_current = 0.0      # initial control input
u_prev = np.array([0.0])  # previous control input (for rate penalty)

# Reference trajectory function for the outputs (cart position and pendulum angle)
def get_reference(t):
    if t < 1.0:
        return np.array([0.0, 0.0])
    else:
        return np.array([1.0, 1.0])

# Data storage for plotting:
time_data = []
state_data = []
control_data = []

# ----------------------------
# Initialize MuJoCo visualization (GLFW)
# ----------------------------
glfw.init()
window = glfw.create_window(1200, 900, "Self-Balancing Robot (Tuned MPC)", None, None)
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
    # Extract current state from MuJoCo:
    state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
    ref = get_reference(t_sim)
    
    # Update control every mpc_update_steps simulation steps:
    if step_count % mpc_update_steps == 0:
        u_current = solve_mpc(state, u_prev, ref, A_d, B_d, N, Nu, P_terminal)
        u_prev = np.array([u_current])
    
    # Apply control input:
    data.ctrl[0] = u_current
    
    # Step the simulation:
    mj.mj_step(model, data)
    
    # Record simulation data:
    time_data.append(t_sim)
    state_data.append(state.copy())
    control_data.append(u_current)
    
    # Render the scene:
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
