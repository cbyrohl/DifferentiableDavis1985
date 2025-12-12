Title: The Evolution of Large-Scale Structure in a Universe Dominated by Cold Dark Matter (Davis, Efstathiou, Frenk, White; ApJ 292:371–394, 1985)

Citation key: 1985ApJ...292..371D

Overview
- Goal: Simulate nonlinear gravitational clustering in CDM-dominated universes and compare clustering statistics with galaxy observations.
- Method: High-resolution N-body with periodic boundary, 32,768 particles, initial conditions from a “constant-curvature” (Harrison–Zel’dovich) primordial spectrum passed through a CDM transfer that bends at horizon entry.
- Cosmologies: Primarily Omega_m=1 (flat) CDM; also Omega_m≈0.2 open-like case; and one flat model with positive Lambda.
- Key statistics: Two-point correlation xi(r), three-point correlation zeta (hierarchical amplitude Q), pairwise peculiar velocity dispersion, cluster mass distribution, visual morphology (filaments, superclusters, voids).

Core Claims (from abstract)
- Morphology: Simulations produce large filamentary structures and large low-density regions, though not as extreme as the Bootes void.
- Two-point correlation: xi(r) evolution is not self-similar; effective power-law slope steepens (more negative) with time.
- Unbiased Omega_m=1 tension: If galaxies trace mass, predicted peculiar velocities are too large; when xi(r) shape matches observed, amplitude is too low for any reasonable H0.
- Lower density: Omega_m≈0.2 improves agreement but still mismatches pairwise velocity scaling with separation.
- Three-point: zeta follows hierarchical form with scale-dependent Q; on small scales Q exceeds observed values.
- Lambda model: A flat model with positive Lambda resembles an open model with same Omega_m.
- Biasing solution: If galaxies form at rare peaks (biased tracers), two- and three-point statistics and relative velocities can match observations even with Omega_m=1.

Simulation Setup (as described)
- N-body: Periodic box; N=32,768 particles; resolves 16× range of length scales in ICs.
- ICs: Adiabatic, scale-invariant primordial spectrum (n≈1). CDM transfer introduces a turnover at horizon-entry scale.
- Runs: Ensembles at Omega_m=1; a case with Omega_m≈0.2; a flat Lambda>0 case. Evolution tracked across epochs; statistics computed vs. observations available circa mid-1980s.

Statistics and Definitions
- Two-point correlation: xi(r) = <delta(x) delta(x+r)>, fitted as ~ (r/r0)^(-gamma) with time-evolving r0, gamma.
- Three-point correlation: zeta(r1,r2,r3) ≈ Q [xi(r12)xi(r23) + xi(r23)xi(r31) + xi(r31)xi(r12)], with Q weakly scale dependent; small-scale Q larger than observed.
- Pairwise velocities: rms relative peculiar velocity as a function of pair separation; simulations predict decrease with separation, whereas observed increases slowly.

Comparison to Observations (per abstract)
- Unbiased Omega_m=1: too-large peculiar velocities; mismatched xi amplitude vs H0 constraints.
- Omega_m≈0.2: better but still tension in velocity–separation trend.
- Peak biasing: rare-peak selection can reconcile clustering and velocities with observations even for Omega_m=1.

Reproduction Targets (minimum viable)
- IC generation: Create Gaussian random field with primordial n=1 spectrum modulated by a CDM transfer function with a turnover; normalize to target sigma8 or xi(r) at a chosen epoch.
- Gravity solver: Periodic N-body (PM/TreePM/fast multipole); softening and timestep control adequate to capture small-scale clustering without artificial collisionality.
- Cosmology grid: Omega_m in {1.0, 0.2}; optionally a flat LambdaCDM with matching Omega_m.
- Measurements: xi(r) over a range of scales; zeta and hierarchical Q on triangle configurations; pairwise velocity dispersion vs separation; mass function proxy (friends-of-friends) for cluster breadth.
- Morphology: Visual diagnostics of filamentary structure and voids for qualitative comparison.

Modernized Implementation Notes
- Transfer function: Use an analytic CDM fit (e.g., BBKS or Eisenstein–Hu no-wiggle) to emulate the bend/turnover. Parameter Gamma≈Omega_m h controls the turnover scale.
- Normalization: Because the paper discusses tensions tied to H0 and amplitudes, run variants with different sigma8 or growth-normalized epochs to reproduce “matching shape but low amplitude” scenarios.
- Bias model: Implement simple peak/threshold bias (select halos/particles above initial high-peak threshold) to reproduce “biased galaxy” comparisons of two- and three-point statistics and velocities.
- Box and resolution: Choose box size and particle number to resolve at least ~1–2 decades in scale for xi(r); periodic boundaries and CIC/TSC mass assignment for grid-based statistics.

Evaluation Checklist
- xi(r): Recover non-self-similar evolution; quantify gamma(z) trend and r0(z).
- Pairwise velocity: Confirm simulated rms decreases with separation for mass tracers; test if biased tracers flatten/increase toward observed trend.
- zeta and Q: Measure Q(scale); verify excess on small scales for mass; test improvement with biasing.
- Morphology: Qualitatively recover filaments, superclusters, voids; note lack of extreme Bootes-like voids without special conditions.
- Lambda vs open: Show degeneracy between flat+Lambda and open with same Omega_m at fixed growth stage.

Data/Outputs to Produce
- Power spectra and correlation functions at several scale factors.
- Pairwise velocity statistics and histograms across separations.
- Triangle-configuration three-point estimates and Q vs scale.
- Friends-of-friends catalogs and mass distributions.
- Slices/renders of density field for qualitative figures.

Open Items / Ambiguities
- Exact transfer function used in 1985 runs is not specified in the abstract; we will adopt a standard CDM fit (BBKS/Eisenstein–Hu) and document differences.
- Observational baselines (surveys, exact xi fits, velocity measurements) should be matched approximately; we will focus on qualitative/relative agreements rather than exact 1985 datasets.
- Normalization/H0: The abstract references amplitude vs H0 tension; we will present outcomes across a range of normalizations to illustrate this trade-off.

Next Steps
- Implement IC generator with configurable n, transfer, and sigma8.
- Run a small pilot N-body to validate statistics pipeline (xi, zeta, velocities).
- Scale to production runs for Omega_m in {1.0, 0.2} and one flat Lambda model.
- Add simple peak-bias selection and remeasure statistics.

