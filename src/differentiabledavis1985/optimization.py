"""
Optimization functions for reconstructing initial conditions from target fields.

This module provides tools for solving inverse problems:
- Given a target final density field, reconstruct the initial conditions
- Uses gradient-based optimization with JAX automatic differentiation
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Tuple, Optional

from .forward_model import Davis1985Simulation
from .losses import chi2, chi2_log


def density_reconstruction_loss(
    initial_conditions: jnp.ndarray,
    target_density: jnp.ndarray,
    model: Davis1985Simulation,
    loss_type: str = "chi2",
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> float:
    """
    Loss function for density field reconstruction.

    This function runs the forward simulation from initial conditions,
    paints the final density field, and compares it to the target.

    Args:
        initial_conditions: Current ICs being optimized (first arg for jax.grad)
        target_density: Target final density field
        model: Simulation model
        loss_type: Type of loss ("chi2" or "chi2_log")
        rtol: Relative tolerance for ODE integrator (default 1e-4, relaxed for speed)
        atol: Absolute tolerance for ODE integrator (default 1e-4, relaxed for speed)

    Returns:
        Loss value (scalar)
    """
    # Run forward simulation with relaxed tolerances for speed
    final_positions, _ = model.run_simulation(initial_conditions, rtol=rtol, atol=atol)

    # Paint final density field (last snapshot)
    predicted_density = model.paint_density(final_positions[-1])

    # Compute loss
    if loss_type == "chi2":
        return chi2(predicted_density, target_density)
    elif loss_type == "chi2_log":
        return chi2_log(predicted_density, target_density)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def optimize_initial_conditions(
    target_density: jnp.ndarray,
    model: Davis1985Simulation,
    loss_fn: Optional[Callable] = None,
    n_iterations: int = 200,
    learning_rate: float = 1e-1,
    seed: int = 1234,
    print_every: int = 10,
    loss_type: str = "chi2",
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize initial conditions to match target final density.

    Uses Adam optimizer with JAX automatic differentiation to solve
    the inverse problem: find ICs that produce the target final density.

    Args:
        target_density: Target final density field
        model: Davis1985Simulation instance
        loss_fn: Custom loss function(params, target, model) -> scalar.
                 If None, uses density_reconstruction_loss with loss_type.
        n_iterations: Number of optimization iterations
        learning_rate: Adam optimizer learning rate
        seed: Random seed for initialization
        print_every: Print loss every N iterations
        loss_type: Type of loss ("chi2" or "chi2_log"), used if loss_fn is None
        rtol: Relative tolerance for ODE integrator (default 1e-4 for speed)
        atol: Absolute tolerance for ODE integrator (default 1e-4 for speed)

    Returns:
        Tuple of (optimized_ics, loss_history)
        - optimized_ics: Optimized initial conditions
        - loss_history: Array of loss values during optimization
    """
    print(f"Starting optimization:")
    print(f"  Target shape: {target_density.shape}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss type: {loss_type}")
    print(f"  ODE tolerances: rtol={rtol}, atol={atol}")
    print()

    # Use default loss function if none provided
    if loss_fn is None:
        def loss_fn(ics, target, mdl):
            return density_reconstruction_loss(ics, target, mdl, loss_type=loss_type, rtol=rtol, atol=atol)

    # Initialize from random ICs (scaled down by 0.1 for better convergence)
    print(f"Generating initial conditions with seed {seed}...")
    params = 0.1 * model.generate_initial_conditions(seed=seed)

    # Set up optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Create gradient function (differentiates w.r.t. first argument)
    grad_fn = jax.grad(loss_fn)

    # Optimization loop
    losses = []
    print("Starting optimization loop...", flush=True)
    print("NOTE: First iteration may take 30-60 seconds due to JAX compilation.", flush=True)
    print(f"{'Iteration':>10} | {'Loss':>12} | {'Time (s)':>10}", flush=True)
    print("-" * 38, flush=True)

    import time
    start_time = time.time()
    iteration_times = []

    for iteration in range(n_iterations):
        iter_start = time.time()

        # Compute gradients
        gradients = grad_fn(params, target_density, model)

        # Update parameters
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        # Track progress - print first few iterations to show it's working
        should_print = (iteration < 3) or (iteration % print_every == 0) or (iteration == n_iterations - 1)

        if should_print:
            current_loss = loss_fn(params, target_density, model)
            losses.append(float(current_loss))
            elapsed = time.time() - start_time
            print(f"{iteration:>10} | {current_loss:>12.6e} | {elapsed:>10.2f}", flush=True)

            # After first iteration, estimate remaining time
            if iteration == 0:
                avg_iter_time = iter_time
                est_total = avg_iter_time * n_iterations
                print(f"  -> First iteration took {iter_time:.1f}s. Estimated total: {est_total:.1f}s (~{est_total/60:.1f} min)", flush=True)

    print()
    print("Optimization completed!")
    return params, jnp.array(losses)


def generate_target_and_reconstruct(
    model: Davis1985Simulation,
    target_seed: int = 42,
    reconstruction_seed: int = 1234,
    n_iterations: int = 200,
    learning_rate: float = 1e-1,
    loss_type: str = "chi2",
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> dict:
    """
    Generate a target density field and reconstruct it (for testing).

    This is a convenience function that:
    1. Generates ICs with target_seed
    2. Runs forward simulation to get target density
    3. Reconstructs ICs from target density using different seed
    4. Returns all intermediate results for comparison

    Args:
        model: Davis1985Simulation instance
        target_seed: Seed for generating target ICs
        reconstruction_seed: Seed for initializing reconstruction
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimizer
        loss_type: Type of loss function
        rtol: Relative tolerance for ODE integrator
        atol: Absolute tolerance for ODE integrator

    Returns:
        Dictionary with:
        - 'target_ics': Original initial conditions
        - 'target_density': Target final density field
        - 'reconstructed_ics': Reconstructed initial conditions
        - 'reconstructed_density': Density from reconstructed ICs
        - 'losses': Loss history during optimization
    """
    print("=" * 60)
    print("GENERATING TARGET")
    print("=" * 60)

    # Generate target ICs
    target_ics = model.generate_initial_conditions(seed=target_seed)

    # Run forward simulation
    print(f"Running forward simulation with seed {target_seed}...")
    target_positions, _ = model.run_simulation(target_ics, rtol=rtol, atol=atol)

    # Paint target density
    target_density = model.paint_density(target_positions[-1])
    print(f"Target density stats: min={target_density.min():.2e}, "
          f"mean={target_density.mean():.2e}, max={target_density.max():.2e}")
    print()

    print("=" * 60)
    print("RECONSTRUCTING")
    print("=" * 60)

    # Reconstruct
    reconstructed_ics, losses = optimize_initial_conditions(
        target_density=target_density,
        model=model,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        seed=reconstruction_seed,
        loss_type=loss_type,
        rtol=rtol,
        atol=atol
    )

    # Run forward simulation with reconstructed ICs
    print("Running forward simulation with reconstructed ICs...")
    reconstructed_positions, _ = model.run_simulation(reconstructed_ics, rtol=rtol, atol=atol)
    reconstructed_density = model.paint_density(reconstructed_positions[-1])
    print(f"Reconstructed density stats: min={reconstructed_density.min():.2e}, "
          f"mean={reconstructed_density.mean():.2e}, max={reconstructed_density.max():.2e}")
    print()

    return {
        'target_ics': target_ics,
        'target_density': target_density,
        'reconstructed_ics': reconstructed_ics,
        'reconstructed_density': reconstructed_density,
        'losses': losses,
    }
