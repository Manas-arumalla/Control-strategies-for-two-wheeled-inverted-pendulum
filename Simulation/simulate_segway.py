from collections import deque
import time
import mujoco
import numpy as np
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
import control as ctrl

# Setup the OpenGL format for high-DPI displays etc.
fmt = QSurfaceFormat()
fmt.setDepthBufferSize(24)
fmt.setStencilBufferSize(8)
fmt.setSamples(4)
fmt.setSwapInterval(1)
fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
fmt.setVersion(2, 0)
fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(fmt)


class Viewport(QOpenGLWindow):
    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()
        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scn = scn
        self.width = 0
        self.height = 0
        self.scale = 1.0
        self.__last_pos = None
        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60 * 1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def setScreenScale(self, scaleFactor: float) -> None:
        self.scale = scaleFactor

    def paintGL(self) -> None:
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)
        self.runtime.append(time.time() - t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True
        self.speed = 0.0  # This will be updated via the speed slider
        self.reset()

        # --- LQR Controller Setup for the segway model ---
        # Physical parameters (from your MATLAB derivation)
        mw = 0.432         # mass of each wheel [kg]
        mp = 5.0           # mass of the pendulum [kg]
        Iw = 0.5 * mw * (0.0726**2)  # moment of inertia of each wheel
        Ip = 5.0 * (0.4**2)          # moment of inertia of the pendulum
        l = 0.4            # pendulum length [m]
        r = 0.0726         # wheel radius [m]
        g = 9.81           # gravitational acceleration [m/s^2]

        # Derived parameters
        a = 2 * mw + mp + 2 * Iw / (r**2)
        b = mp * l
        c = mp * (l**2) + Ip
        d = mp * g * l
        Delta = a * c - b**2

        # Construct continuous-time state-space matrices.
        # State: [x, x_dot, theta, theta_dot]
        A = np.array([
            [0,      1,           0,           0],
            [0,      0,   -b * d / Delta,       0],
            [0,      0,           0,           1],
            [0,      0,    a * d / Delta,       0]
        ])
        B = np.array([[0],
                      [c / Delta],
                      [0],
                      [-b / Delta]])
        Q = np.diag([100, 1, 1000, 1])
        R = np.array([[0.01]])
        K, S, E = ctrl.lqr(A, B, Q, R)
        self.K = np.asarray(K)
        print("LQR Gain K:", self.K)

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self) -> None:
        control_interval = 1/200  # update control at 200 Hz
        while self.running:
            # Keep simulation from running ahead of real time
            if self.data.time < self.real_time / 1_000_000_000:
                current_time = time.monotonic_ns()
                if (current_time - self.last_control_update) / 1_000_000_000 >= control_interval:
                    self.last_control_update = current_time
                    # Extract current state:
                    # qpos[0]: slider position (x)
                    # qvel[0]: slider velocity (x_dot)
                    # qpos[1]: pendulum angle (theta)
                    # qvel[1]: pendulum angular velocity (theta_dot)
                    state = np.array([
                        self.data.qpos[0],
                        self.data.qvel[0],
                        self.data.qpos[1],
                        self.data.qvel[1]
                    ])
                    # Compute control: we use the LQR feedback and add a feedforward term from the speed slider.
                    u = -np.dot(self.K, state) + self.speed
                    self.data.ctrl[0] = u.item()
                mujoco.mj_step(self.model, self.data)
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_control_update = time.monotonic_ns()
        # Reset simulation state.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_speed(self, speed: float) -> None:
        self.speed = speed


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # Load the segway model (ensure segway.xml is in the same directory)
        model_path = pathlib.Path(__file__).parent.joinpath('segway.xml')
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        layout_top = QHBoxLayout()
        layout_top.setSpacing(8)
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(90)
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        reset_button.clicked.connect(self.reset_simulation)
        layout_top.addWidget(reset_button)
        # Only the speed slider is added; yaw slider has been removed.
        layout_robot_controls = QVBoxLayout()
        layout_robot_controls.setContentsMargins(0, 0, 0, 0)
        layout_robot_controls.addWidget(self.create_top())
        layout_top.addLayout(layout_robot_controls)
        layout_top.setContentsMargins(8, 0, 8, 0)
        layout.addLayout(layout_top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setStretch(1, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(800, 600)

        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {self.data.time:.0f}s"
        )

    def create_top(self):
        layout = QVBoxLayout()
        label_width = 60

        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        # The slider range (scaled by 1000) can be adjusted; here we allow a feedforward command from -5 to 5.
        self.speed_slider.setMinimum(-5 * 1000)
        self.speed_slider.setMaximum(5 * 1000)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        speed_label = QLabel("Speed")
        speed_label.setFixedWidth(label_width)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        layout.addLayout(speed_layout)

        group = QGroupBox("Segway Control")
        group.setLayout(layout)
        return group

    def _speed_changed(self, value: int) -> None:
        # Convert the slider integer value back to float
        speed = value / 1000.0
        self.th.set_speed(speed)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([0.0, 0.0, 0.0])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        self.speed_slider.setValue(0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
