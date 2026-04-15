# Quantum-GAN-Calibration: Advanced Pulse Optimization.

An industrial-grade framework for calibrating quantum gates using Physics-Informed Generative Adversarial Networks (PI-GANs). This project uses a GAN architecture to synthesize optimal control pulses that compensate for environmental decoherence and hardware drift.

##  Overview

Quantum calibration usually requires exhaustive sweeps (Rabi/Ramsey). This framework uses:
- **Generator**: A deep neural network that takes target gate parameters and outputs optimal voltage/phase pulses.
- **Discriminator**: A critic that evaluates the "quantumness" of the resulting state, penalizing non-physical transitions and decoherence.
- **Physics Engine**: A Lindblad Master Equation solver integrated into the loss function to ensure gradient flow through physical constraints.

##  Structure

- `/core`: Schrödinger and Lindblad solvers for pulse-to-state mapping.
- `/models`: GAN architectures (Generator/Discriminator/Critic).
- `/training`: Adversarial training loops with physics-informed regularization.
- `/utils`: Bloch sphere visualization and FFT signal analysis.

##  Features

- **Decoherence Modeling**: Simulates T1 and T2 relaxation during calibration.
- **WGAN-GP Implementation**: Uses Wasserstein GAN with Gradient Penalty for stable convergence.
- **Custom Pulse Envelopes**: Supports Gaussian, DRAG, and Square pulses.

##  Prerequisites

See `requirements.txt` for the full list (standard scientific stack: NumPy, SciPy, PyTorch/TensorFlow).

---
*Developed for advanced quantum control research.*
