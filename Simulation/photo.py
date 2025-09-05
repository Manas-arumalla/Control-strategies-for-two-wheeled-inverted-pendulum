import time
import mujoco
import mujoco.viewer
import imageio

# Define your segway MJCF model as a string
mjcf_xml = r"""
<mujoco model="segway">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  
  <!-- Default settings -->
  <default>
    <!-- All geoms default to a light gray color -->
    <geom rgba="0.8 0.8 0.8 1"/>
  </default>
  
  <asset>
    <!-- Skybox with a bright gradient: top is pure white, bottom is near white -->
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.95 0.95 0.95" width="512" height="512"/>
    
    <!-- Ground texture using a bright checker pattern -->
    <texture name="checker" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="1 1 1" width="512" height="512"/>
    <material name="matplane" texture="checker" texrepeat="10 10" texuniform="true" reflectance="0.3"/>
  </asset>
  
  <worldbody>
    <!-- Light source -->
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.5 0.5 0.5" directional="true"/>
    
    <!-- Ground plane -->
    <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1" material="matplane"/>
    
    <!-- Base representing the robot's body -->
    <body name="base" pos="0 0 0.14">
      <!-- Slider joint for horizontal translation (x-axis) -->
      <joint name="slide" type="slide" axis="1 0 0" pos="0 0 0" limited="true" range="-5 5"/>
      
      <!-- Two wheels with reduced lateral separation -->
      <!-- Left wheel (positioned at y = 0.1 relative to base) -->
      <geom name="wheel_left" type="cylinder" pos="0 0.1 -0.07" size="0.07 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      <!-- Right wheel (positioned at y = -0.1 relative to base) -->
      <geom name="wheel_right" type="cylinder" pos="0 -0.1 -0.07" size="0.07 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      
      <!-- Inverted pendulum attached to the base -->
      <!-- Lowered slightly (pos="0 0 -0.05") to reduce gap -->
      <body name="pendulum" pos="0 0 -0.05">
        <!-- Hinge joint for pendulum tilt (rotates about y-axis) -->
        <joint name="pend_hinge" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.1"/>
        <!-- Pendulum rod as a capsule extending upward -->
        <geom name="pendulum_geom" type="capsule" fromto="0 0 0 0 0 0.4" size="0.03" rgba="0.8 0.2 0.2 1"/>
        <site name="pend_tip" pos="0 0 0.4" size="0.03" rgba="0 1 0 1"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Actuator applies force to the slider joint -->
    <general name="drive" joint="slide" gear="1" ctrllimited="true" ctrlrange="-50 50"/>
  </actuator>
  
  <sensor>
    <jointpos name="slide_pos" joint="slide"/>
    <jointvel name="slide_vel" joint="slide"/>
    <jointpos name="pend_angle" joint="pend_hinge"/>
    <jointvel name="pend_vel" joint="pend_hinge"/>
  </sensor>
</mujoco>


"""

# Create the model and data objects
model = mujoco.MjModel.from_xml_string(mjcf_xml)
data = mujoco.MjData(model)

# Launch an interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while time.time() - start_time < 100:
        mujoco.mj_step(model, data)
        viewer.sync()  # update viewer display
        time.sleep(model.opt.timestep)
    
