# Control Strategies for Two-Wheeled Inverted Pendulum

This repository presents a comprehensive study and implementation of various control strategies for a two-wheeled inverted pendulum system—akin to a Segway. The project aims to evaluate and compare multiple advanced control methods to achieve robust stability and performance under diverse operating conditions.

## Implemented Controllers

- **Linear Quadratic Regulator (LQR):**  
  Optimal state feedback control minimizing a quadratic cost function.

- **Pole Placement:**  
  Custom controller design by assigning desired closed-loop pole locations.

- **H∞ Control:**  
  Robust control strategy to attenuate the effect of disturbances and model uncertainties.

- **Model Predictive Control (MPC):**  
  Advanced control leveraging online optimization over a finite horizon.

- **Sliding Mode Control:**  
  Control ensures robust performance by driving system states to a sliding surface.

- **Hybrid LQR + Sliding Mode + Backstepping:**  
  Combines optimal control, robustness, and nonlinear design techniques.

- **LQR + L1 Adaptive Controller:**  
  Integrates adaptive elements with LQR for improved real-time disturbance rejection.

- **Nonlinear Dynamic Inversion (NDI) with PD Controller:**  
  Utilizes system inversion coupled with proportional-derivative feedback.

- **Carleman Linearization + LQR:**  
  Applies Carleman linearization to nonlinear dynamics, then stabilizes with LQR.

- **Linear Quadratic Virulence (LQV):**  
  A variant of LQR targeting performance under uncertain conditions.

- **EPSAC:**  
  Economic predictive control strategy aimed at optimizing performance and efficiency.

- **DSC + Neural Network (NN) + Nonlinear MPC (NMPC):**  
  Integrates dynamic surface control, learning-based approaches, and nonlinear predictive control for enhanced adaptability.

## Genetic Algorithm-Based Parameter Optimization

All controller parameters have been finely tuned using a genetic algorithm to ensure the best possible performance from each control strategy. This optimization systematically explores the parameter space to enhance stability, response time, and overall system performance.

## Project Highlights

- **Comparative Analysis:**  
  Detailed simulation and experimental results comparing performance, robustness, and computational complexity across controllers.

- **Design Insights:**  
  In-depth guidance on controller tuning, stability margins, and advanced control system design trade-offs.

- **Educational Resource:**  
  A reference for researchers and practitioners interested in cutting-edge control techniques and optimization strategies for nonlinear and uncertain systems.
