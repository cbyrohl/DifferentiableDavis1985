# Reproduction Data: Davis, Efstathiou, Frenk, & White (1985)

**Source:** The Astrophysical Journal, 292:371-394, 1985 May 15
**Objective:** Reproduce particle distributions shown in Figure 1.

## 1. Simulation Methodology

### Numerical Code
*   **Type:** P³M (Particle-Particle / Particle-Mesh).
*   **Grid:** $64^3$ cells for Fourier force calculation.
*   **Particles ($N$):** 32,768 ($32^3$).
*   **Boundary Conditions:** Triply periodic.
*   **Time Integration:** Leapfrog scheme using expansion parameter $a$ as the time variable.
*   **Timestep:** $\Delta a = 0.02$.
*   **Force Softening:** Short-range force correction applied. Softening length $\eta = L/213$ (where $L$ is comoving box side).

### Initial Conditions (ICs)
*   **Method:** Zel'dovich (1970) approximation.
*   **Setup:** Particles start on a cubic lattice.
*   **Perturbations:** Displacements and velocities assigned proportional to the gradient of the potential derived from a random realization of the linear fluctuation distribution.
*   **Phasing:** Random phases; amplitudes distributed normally.
*   **Sampling Limit:** Frequencies correspond to waves with projected wavelength $> L/16$ (Nyquist frequency of the particle grid).

## 2. Cosmological Model & Power Spectrum

### Spectrum Formulation
The Cold Dark Matter (CDM) power spectrum is approximated by:
$$ |\delta_k|^2 = \frac{Ak}{ (1 + \alpha k + \beta k^{3/2} + \gamma k^2)^2 } $$

### Parameters
*   $l = (\Omega h^2 \theta^{-2})^{-1}$ Mpc
*   $\alpha = 1.7 l$
*   $\beta = 9.0 l^{3/2}$
*   $\gamma = 1.0 l^2$
*   $\theta = 1$ (microwave background temperature ratio).

### Normalization
*   The amplitude $A$ is chosen such that the **theoretical power equals the white-noise level at a wavelength of $L/16$**.
*   White noise level reference: Defined by the discrete particle shot noise.

## 3. Specific Models (Figure 1)

Figure 1 displays 2D projections of particle positions from two specific runs: **EdS1** and **O1**.

### Model EdS1 (Einstein-de Sitter)
*   **$\Omega$:** 1.0
*   **Box Size ($L$):** $32.5 h^{-2}$ Mpc.
*   **Initial Epoch:** $a = 1.0$.

### Model O1 (Open Universe)
*   **$\Omega$:** Evolves. $\Omega = 1.0$ initially, $\Omega = 0.2$ at $a = 3.2$.
*   **Box Size ($L$):** Equivalent to EdS1 but scaled. Mass reduced. Defined such that at $a=3.2$, $L$ represents $162.5 h^{-2}$ Mpc.
*   **Spectrum Fit:** Parameters adjusted to fit Blumenthal & Primack (1983) results for $\Omega=0.2$.

## 4. Figure 1 Plotting Specifications

**Visual Style:** 2D Projections (Full box depth projected onto a plane). Points represent individual dark matter particles.

### Panel Configuration (6 Plots)

**Left Column (Model EdS1 - $\Omega=1$):**
1.  **Top Left:** Initial Conditions ($a = 1.0$).
2.  **Center Left:** Evolution at $a = 1.8$ (Linear growth factor = 1.8).
3.  **Bottom Left:** Evolution at $a = 2.4$ (Linear growth factor = 2.4).
    *   *Note:* Shows filamentary structures and superclusters.

**Right Column (Model EdS1 & O1):**
4.  **Top Right:** Model **EdS1** at $a = 4.5$.
    *   *Note:* dominated by large clumps, clustering scale approaches box size.
5.  **Center Right:** Model **O1** at $a = 3.2$.
    *   *State:* $\Omega = 0.2$.
    *   *Matching Condition:* Matched to Center Left (EdS1) by linear growth factor (1.8).
6.  **Bottom Right:** Model **O1** at $a = 8.4$.
    *   *State:* $\Omega = 0.09$.
    *   *Matching Condition:* Matched to Bottom Left (EdS1) by linear growth factor (2.4).
