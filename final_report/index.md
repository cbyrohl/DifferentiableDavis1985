# Differentiable Davis 1985 - Final Report

## JaxPM Forward Simulation

Initial (z=99) and final (z=2) density fields from PM N-body evolution.

![Overdensity](../output/overdensity_validation.png)

## Time Evolution

Gravitational collapse forming filaments and voids.

![Time Evolution](../output/demonstration/time_evolution.gif)

## Optimization

Gradient-based reconstruction of initial conditions from final density.

![Reconstruction Animation](../output/reconstruction/reconstruction_animation.gif)

## Reconstruction

Target: Figure 1 (lower left) from Davis et al. (1985).

![Original Figure](../paper_material/fig1_lowerleft_original.png)

2D projection mask extracted from the original scatter plot.

![Binned Overdensity](../output/mask_binned/binned_overdensity.png)

Reconstructed density distribution via gradient descent on 2D projection loss.

![2D Reconstruction](../output/reconstruction_2d_32/reconstruction_animation_2d.gif)

