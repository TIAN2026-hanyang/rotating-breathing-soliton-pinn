# Rotating, Breathing Soliton
Analytical-prior-assisted physics-informed neural network for rotating-breathing solitons in strongly nonlocal media

## Explanation
Directly approximating multidimensional, moving, complex fields using a physics-informed neural network is computationally expensive because the network must simultaneously handle translation, radial breathing, angular rotation, phase evolution, and nonlinear field corrections.
We propose an analytical-prior-assisted physics-informed neural network for analyzing rotating, breathing soliton molecules governed by the 2D nonlinear Schrödinger equation in strongly nonlocal media.

The exact solution for the \(w=0\) reference linear model is implemented as the spatiotemporal field backbone. The neural network only learns the nonlinear correction caused by the weak local Kerr term (\(w=0.02\)).
Combined with a precise initial condition embedding and a two-stage optimization strategy, this method significantly reduces training costs. The Split-Step Fourier Method (SSFM) is only used for post-training validation and is not involved in the training process.

We tested inward-breathing, near-constant-radius, outward-breathing dynamical regimes for 4-component and 6-component soliton molecules. The results demonstrate that embedding a known reference propagation at the field-representation level can convert direct full-field PINN approximation into a computationally feasible method.
efficient learning when the target dynamics remain sufficiently close to a solvable solution
reference model

## 🛠️ Environment & Dependencies
The code is implemented using **PyTorch**. It has been tested on an NVIDIA RTX 4070 SUPER GPU.

- Python == 3.11.14
- torch version == 2.9.1
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install torch==2.9.4 numpy scipy matplotlib
