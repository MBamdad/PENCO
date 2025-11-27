# PENCO: Physics–Energy–Numerical–Consistent Operator for 3D Phase-Field Modeling

This repository contains the official implementation of **PENCO** (Physics–Energy–Numerical–Consistent Operator), a hybrid neural operator framework for **three-dimensional phase-field PDEs**.

PENCO combines:
- **Neural operators** (spectral/Fourier-based),
- **Physics constraints** (PDE residuals evaluated at symmetric Gauss–Lobatto collocation points),
- **Numerical consistency** (alignment with a stable semi-implicit update),
- **Thermodynamic energy dissipation** (enforcing one-sided free-energy decay),

to achieve **accurate, stable, and physically consistent** long-horizon predictions with **very limited training data**.

---

## 1. Problem Setting

We consider phase-field-type PDEs of the form

> ∂ₜu = 𝒩(u, ∇u, ∇²u, …),

where:
- **u(x, t)** is a scalar order parameter on a periodic 3D domain Ω ⊂ ℝ³,
- 𝒩 is a nonlinear differential operator (e.g., Allen–Cahn, Cahn–Hilliard, etc.).

The goal is to learn a **solution operator**

> 𝒢\_θ : a(x) ↦ u(x, t),

that maps an initial condition **a(x)** to the full spatio-temporal trajectory **u(x, t)**.

Supervision is provided by a small number of **high-fidelity spectral simulations** (semi-implicit Fourier-spectral solver) for several benchmark PDEs:
- Allen–Cahn (AC),
- Cahn–Hilliard (CH),
- Swift–Hohenberg (SH),
- Phase-Field Crystal (PFC),
- Molecular Beam Epitaxy (MBE).

---

## 2. Baseline Neural Operators

We benchmark PENCO against two state-of-the-art neural operator architectures.

### 2.1 Four-Dimensional Fourier Neural Operator (FNO-4D)

- Learns mappings between function spaces using **spectral convolutions** in space.
- Time is treated as an additional input channel; Fourier transform is applied only in spatial dimensions (x, y, z).
- Architecture structure:
  - Lifting network 𝒫(a(x, t)) → latent representation v₀(x, t),
  - Sequence of spectral layers combining pointwise linear maps and Fourier convolutions,
  - Projection network 𝒬 that maps latent features back to physical space.
- Predicts blocks of time steps in a **single forward pass**, avoiding explicit recurrence.

**Limitations:**
- No explicit causal structure over time,
- Temporal errors accumulate over long horizons,
- No explicit enforcement of physics or numerical structure.

### 2.2 Multi-Head Neural Operator (MHNO)

- Uses a **shared spectral encoder** for spatial features and **time-step-specific heads** for temporal evolution.
- Components:
  - Spectral encoder: shared across all times, maps the initial field to a high-dimensional latent representation.
  - Projection networks 𝒬ₙ: time-dependent decoders for each time level.
  - Transition networks ℋₙ: propagate information between successive time steps.

**Advantages:**
- Captures long-range temporal dependencies,
- Improves temporal coherence over purely parallel or purely autoregressive designs.

**Limitations:**
- Still trained with a purely data-driven loss (mean-squared error),
- No explicit guarantee of energy decay or numerical scheme consistency.

---

## 3. PENCO: Physics–Energy–Numerical–Consistent Operator

PENCO keeps a compact **spectral neural operator backbone** but fundamentally changes the **training objective**.

Instead of only minimizing a data regression loss, PENCO optimizes

> 𝓛\_total = (1 − λ) · α · 𝓛\_data + λ · 𝓛\_phys,  with λ ∈ [0, 1],

where:
- **𝓛_data**: standard prediction error against the ground-truth solution, measured using the MSE formulation.
- **𝓛_phys**: physics-guided regularization composed of:
  - **PDE residuals** at symmetric Gauss–Lobatto collocation points,
  - **Numerical scheme consistency** via a semi-implicit reference update,
  - **Energy dissipation** through one-sided free-energy decay enforcement,
  - **Low-frequency spectral anchoring** to stabilize large-scale modes.

A typical choice:
- α = 10³,
- λ = 0.25 for hybrid training,
- λ = 1.0 for pure-physics training.

### 𝓛_colloc (PDE collocation residual)

Enforces the PDE inside each time step using L² Gauss–Lobatto collocation.  
The residual is evaluated at two temporal points:

**τ₁,₂ = 1/2 ± 1/(2√5)**

At each τ, the PDE residual is computed as:

- the predicted time derivative  
  **(ûⁿ⁺¹ − uⁿ) / Δt**
- minus the PDE right-hand side evaluated at the interpolated state  
  **(1 − τ)·uⁿ + τ·ûⁿ⁺¹**
  
The two residuals are normalized and combined through their L² norm to form the collocation loss.


- **𝓛\_scheme (numerical scheme consistency)**  
  Aligns the network update with a **semi-implicit IMEX time-stepping scheme**:
  - Linear stiff terms treated implicitly,
  - Nonlinear terms treated explicitly,
  - Mobility operator encodes gradient flow geometry.  
  The neural prediction is penalized if it deviates from the semi-implicit update.

- **𝓛\_energy (thermodynamic consistency)**  
  Penalizes any **increase in free energy** between successive time steps:
  - E[uⁿ⁺¹] ≤ E[uⁿ] enforced via a one-sided penalty,
  - E[u] includes double-well bulk energy and gradient regularization terms (and higher-order contributions when relevant).

- **𝓛\_anchor (low-frequency spectral anchoring)**  
  Compares the **low-wavenumber Fourier modes** of:
  - predicted field, and
  - semi-implicit teacher.  
  This prevents large-scale spectral drift and maintains coherent long-range structure.

### 3.2 Time-Dependent Weights

- **w₁ = 10⁻³** and **w₃ = 0.3** (fixed throughout training).

- **w₂ and w₄ (epoch-dependent weights)**  
  These weights evolve smoothly over training following:

  ```
  w₂(e) = 0.32 − 0.12 * (e / (E − 1))
  ```

  ```
  w₄(e) = 0.25 + 0.6 * (e / (E − 1))²
  ```

  where **e** is the current epoch and **E** is the total number of epochs.

  - **Early training:** larger w₂ provides stronger emphasis on scheme consistency, improving short-horizon stability.  
  - **Later training:** increasing w₄ strengthens spectral anchoring, reducing long-horizon drift and stabilizing low-frequency modes.

---

## 4. Data Generation and Training Protocol

### 4.1 Data Generation

- Training trajectories: **N ∈ {50, 100, 200}** per PDE.
- Spatial discretization: 3D periodic cube using a **Fourier-spectral** method.
- Time integration: **semi-implicit IMEX** scheme (the same scheme that defines the numerical teacher).
- Initial conditions:
  - Gaussian Random Fields (GRFs) with randomized statistical parameters and thresholds,
  - This yields a wide range of interface geometries, characteristic length scales, and volume fractions.

The numerical simulations and training are implemented in **PyTorch** and run on NVIDIA GPUs, ensuring consistent floating-point behavior between solver and neural operator.

### 4.2 Training Budget

To isolate **data efficiency** from pure computational budget:

- The **test set size is fixed** (50 trajectories per PDE).
- The **number of gradient updates** is kept constant for a given PDE:
  - Larger datasets do *not* receive more optimizer steps.
  - In the pure-physics limit (λ = 1), the effective number of updates is fixed, independent of N.

The operator architecture is intentionally compact:
- 2 spectral layers,
- Moderate latent width,
- ~800k trainable parameters,
- Spatial resolution: 32³,
- Temporal horizon: Nₜ = 100 time steps.

---

## 5. Numerical Experiments

All models (FNO-4D, MHNO, PENCO hybrid, PENCO pure-physics) are evaluated on the unified phase-field system

> ∂ₜu = α₁[∇²u − (1/ε²) f′(u)]  
>       + α₂ ∇²[−ε² ∇²u + f′(u)]  
>       + α₃[−u³ − (1 − ε)u + 2∇²u − ∇⁴u]  
>       + α₄ ∇²[u³ + (1 − ε)u + 2∇²u + ∇⁴u]  
>       + α₅[−ε ∇⁴u − ∇·((1 − |∇u|²) ∇u)],

with coefficients α₁,…,α₅ selecting AC, CH, SH, PFC, or MBE.

Performance is measured via a normalized L² error over space–time:

> Error =  
>  sqrt( Σ\_i |Û(xᵢ, tᵢ) − u(xᵢ, tᵢ)|² ) / sqrt( Σ\_i |u(xᵢ, tᵢ)|² ),

where Û is the model prediction and u is the reference solution.

### 5.1 In-Distribution Performance (GRF Initial Conditions)

Example: 3D **Allen–Cahn** equation:

- The phase field evolves under competition between:
  - a double-well reaction term driving u toward ±1, and
  - a diffusion term that regularizes interfaces.

**Observations:**

- **FNO-4D**:
  - Tracks early-time dynamics,
  - Gradually oversmooths,
  - Small structures disappear,
  - Topology degrades at later times.

- **MHNO**:
  - Better temporal coherence due to time-step-specific heads,
  - Still exhibits noticeable drift on long horizons.

- **PENCO (λ = 0.25)**:
  - Preserves interface sharpness and correct coarsening rate,
  - Maintains morphology consistent with the reference until final time,
  - Strongly suppresses long-horizon error accumulation.

- **Pure-physics (λ = 1)**:
  - Stable and qualitatively correct,
  - Slightly less accurate in fine details compared to hybrid PENCO.


## 🔬 Numerical Examples

Below we present representative 2D and 3D phase-evolution results for each PDE system using the PENCO hybrid operator.  
These animations demonstrate interface sharpening, coarsening behavior, and long-horizon stability achieved by physics–numerical consistent training.

---

## **Allen–Cahn (AC)**

### **2D Phase Evolution**
The 2D contour evolution highlights PENCO’s ability to maintain sharp interfaces, correct curvature-driven smoothing, and consistent coarsening over the full simulation horizon.

![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/AC3d_results/AC3D_2D_contour_evolution.gif)

### **3D Phase Evolution**
The 3D iso-surface evolution captures full volumetric coarsening behavior. PENCO preserves morphology, interface topology, and large-scale structure more accurately than purely data-driven baselines.

<!-- Replace this with your actual 3D GIF path -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/AC3d_results/AC3D_isosurface_evolution.gif)

---

## **Cahn–Hilliard (CH)**

### **2D Phase Evolution**
<!-- Replace this link with the CH 2D GIF file -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/CH3d_results/CH3D_2D_contour_evolution.gif)

### **3D Phase Evolution**
<!-- Replace this link with CH 3D GIF file -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/CH3d_results/CH3D_isosurface_evolution.gif)

---

## **Swift–Hohenberg (SH)**

### **2D Evolution**
<!-- Replace with SH 2D -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/SH3D_resluts/SH3D_2D_contour_evolution.gif)

### **3D Evolution**
<!-- Replace with SH 3D -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/SH3D_resluts/SH3D_isosurface_evolution.gif)

---

## **Phase-Field Crystal (PFC)**

### **2D Evolution**
<!-- Replace with PFC 2D -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/PFC3D_results/PFC3D_2D_contour_evolution.gif)

### **3D Evolution**
<!-- Replace with PFC 3D -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/PFC3D_results/PFC3D_isosurface_evolution.gif)

---

## **Molecular Beam Epitaxy (MBE)**

### **2D Evolution**
<!-- Replace with MBE 2D -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/MBE3d_results/MBE3D_2D_contour_evolution.gif)

### **3D Evolution**
<!-- Replace with MBE 3D -->
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/MBE3d_results/MBE3D_isosurface_evolution.gif)

---


### 5.2 📘 Out-of-Distribution (OOD) Results

To test robustness beyond GRF-based initial conditions, we evaluate on **deterministic geometries**:

- **AC, CH, SH**: smooth spheres
- **PFC**: anisotropic star-shaped interface
- **MBE**: torus configuration

These shapes differ significantly from training data in curvature, interface width, and topology, making them a stringent test of OOD generalization.


**Summary of OOD behavior:**

- **FNO-4D**:
  - Highest error in almost all OOD settings,
  - Error grows rapidly as dynamics depart from GRF statistics.

- **MHNO**:
  - Consistently better than FNO-4D,
  - Captures sharper interfaces and non-GRF structures more faithfully.

- **PENCO (Hybrid)**:
  - Best overall OOD performance across most systems (especially SH, PFC, MBE),
  - Physics-based regularization removes overshoot and stabilizes long-horizon rollouts,
  - Maintains coherent evolution even for highly non-GRF geometries.

- **Pure-Physics**:
  - Particularly strong for CH, SH, and MBE, where conservative structure or curvature-driven coarsening is dominant,
  - Slightly outperforms PENCO in CH due to strict mass conservation encoded in the PDE,
  - Generally less detailed than the hybrid model.

A complete collection of all OOD evaluation plots (spherical, star-shaped, torus initializations across all PDEs) is provided in the PDF below:
![](https://github.com/MBamdad/PENCO/blob/main/AC3D_Hybrid/hybrid_ac3d/OOD_plots-1.png)

---

## 6. Key Takeaways

Compared to FNO-4D and MHNO, PENCO demonstrates:

- **Superior long-horizon stability and accuracy**, enabled by physics-consistent training,
- **High data efficiency**, achieving strong performance with only 50–200 training trajectories,
- **Robust OOD generalization** to geometries entirely absent from the training distribution (spherical, star-shaped, torus initializations),
- **Physically faithful evolution**, consistently preserving energy decay, correct coarsening rates, interface sharpness, and wavelength-selection behavior.

Furthermore, PENCO achieves these improvements with a much lighter computational footprint, including:

- **32³ spatial resolution**, far smaller than typical operator-learning configurations,
- **Only two spectral operator layers**, minimizing model depth while retaining expressiveness,
- **A training horizon reduced by a factor of ten**, achieving stable convergence with only ~10% of the epochs normally required,
- **No loss in accuracy, stability, or generalization**, despite the significantly reduced computational budget.

Together, these factors demonstrate PENCO’s efficiency and scalability for large-scale 3D phase-field modeling.

---

## 7. Citation

If you would like others to cite this work, you can add a BibTeX entry here once the paper is on arXiv or published, for example:

```bibtex
@article{bamdad2025penco,
  title   = {PENCO: A Physics–Energy–Numerical–Consistent Operator for 3D Phase Field Modeling},
  author  = {Bamdad, Mostafa and Eshaghi, Mohammad Sadegh and Anitescu, Cosmin and Valizadeh, Navid and Rabczuk, Timon},
  journal = arXiv,
  year    = {2025},
}
```
