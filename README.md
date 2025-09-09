# Project Description

## Control Strategies for Two-Wheeled Inverted Pendulum

This repository presents a comprehensive study and implementation of various control strategies for a two-wheeled inverted pendulum system—akin to a Segway. The project aims to evaluate and compare multiple advanced control methods to achieve robust stability and performance under diverse operating conditions.

At its core, the system is modelled as a pendulum mounted on a two-wheel base. The dynamics are derived and linearized about the upright position to obtain a 4-state state-space model. Using this model, a wide range of controllers are designed, tuned, and tested in both MATLAB and Python. Simulations are performed to evaluate each controller’s ability to stabilize the pendulum, recover from disturbances, and maintain robustness in the presence of modelling uncertainties. A MuJoCo physics model is also used for realistic testing and visualization.

---

## Implemented Controllers

- **Linear Quadratic Regulator (LQR):** Optimal state feedback control minimizing a quadratic cost function.
  
- **Pole Placement:** Custom controller design by assigning desired closed-loop pole locations.
    
- **H∞ Control:** Robust control strategy to attenuate the effect of disturbances and model uncertainties.
  
- **Model Predictive Control (MPC):** Advanced control leveraging online optimization over a finite horizon.
  
- **Sliding Mode Control (SMC):** Robust performance by driving system states to a sliding surface.
  
- **Hybrid LQR + Sliding Mode + Backstepping:** Combines optimal control, robustness, and nonlinear design techniques.
  
- **LQR + L1 Adaptive Controller:** Integrates adaptive elements with LQR for improved real-time disturbance rejection.
  
- **Nonlinear Dynamic Inversion (NDI) with PD Controller:** Utilizes system inversion coupled with proportional-derivative feedback.
  
- **Carleman Linearization + LQR:** Applies Carleman linearization to nonlinear dynamics, then stabilizes with LQR.
  
- **Linear Quadratic Virulence (LQV):** A variant of LQR targeting performance under uncertain conditions.  
- **EPSAC:** Economic predictive control strategy aimed at optimizing performance and efficiency.
  
- **DSC + Neural Network (NN) + Nonlinear MPC (NMPC):** Integrates dynamic surface control, learning-based approaches, and nonlinear predictive control for enhanced adaptability.  

---

## Genetic Algorithm-Based Parameter Optimization

To ensure fair comparisons, all controller parameters have been finely tuned using a Genetic Algorithm (GA). The GA systematically explores the parameter space (e.g., LQR’s Q and R matrices) to achieve the best trade-off between stability, settling time, overshoot, and robustness. This automated tuning process enhances the performance and reliability of each control strategy across varying scenarios.

---

## Project Highlights

**Comparative Analysis:**  
Detailed simulation and experimental results comparing performance, robustness, and computational complexity across controllers.  

**Design Insights:**  
In-depth guidance on controller tuning, stability margins, and advanced control system design trade-offs.  

**Educational Resource:**  
Serves as a reference for researchers and practitioners exploring cutting-edge control techniques and optimization strategies for nonlinear and uncertain systems.  
