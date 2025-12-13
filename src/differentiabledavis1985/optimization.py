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
from loguru import logger

from .forward_model import Davis1985Simulation
from .losses import chi2, chi2_log
from .projection import (
    density_to_soft_mask,
    mask_loss_bce,
    mask_loss_mse,
    mask_loss_dice,
    mask_loss_focal,
)


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
    atol: float = 1e-4,
    save_iterations: bool = False,
    save_every: int = 1,
    init_scale: float = 0.1,
    noise_init: bool = False,
    noise_sigma: float = 0.001
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
        save_iterations: If True, save intermediate ICs during optimization
        save_every: Save ICs every N iterations (only if save_iterations=True)
        init_scale: Scaling factor for initial random ICs (default 0.1)
        noise_init: If True, use pure Gaussian noise instead of cosmological ICs
        noise_sigma: Standard deviation for Gaussian noise (default 0.001)

    Returns:
        Tuple of (optimized_ics, loss_history, iterations_ics)
        - optimized_ics: Optimized initial conditions
        - loss_history: Array of loss values during optimization
        - iterations_ics: List of ICs at saved iterations (empty if save_iterations=False)
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

    # Initialize from random ICs
    if noise_init:
        # Pure Gaussian noise around zero overdensity
        logger.info(f"Generating Gaussian noise ICs with seed {seed} (sigma={noise_sigma})")
        print(f"Generating Gaussian noise ICs with seed {seed} (sigma={noise_sigma})...")
        key = jax.random.PRNGKey(seed)
        mesh_shape = target_density.shape
        params = noise_sigma * jax.random.normal(key, shape=mesh_shape)
    else:
        # Cosmological ICs (scaled down for better convergence)
        logger.info(f"Generating initial conditions with seed {seed} (scale={init_scale})")
        print(f"Generating initial conditions with seed {seed} (scale={init_scale})...")
        params = init_scale * model.generate_initial_conditions(seed=seed)

    # Set up optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Create gradient function (differentiates w.r.t. first argument)
    grad_fn = jax.grad(loss_fn)

    # Optimization loop
    losses = []
    iterations_ics = []
    print("Starting optimization loop...", flush=True)
    print("NOTE: First iteration may take 30-60 seconds due to JAX compilation.", flush=True)
    print(f"{'Iteration':>10} | {'Loss':>12} | {'Time (s)':>10}", flush=True)
    print("-" * 38, flush=True)

    import time
    start_time = time.time()
    iteration_times = []

    # Save initial state BEFORE any optimization
    if save_iterations:
        iterations_ics.append((0, params.copy()))

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

        # Save intermediate ICs if requested (iteration+1 since we saved initial as 0)
        if save_iterations and ((iteration + 1) % save_every == 0 or iteration == n_iterations - 1):
            iterations_ics.append((iteration + 1, params.copy()))

    print()
    print("Optimization completed!")
    return params, jnp.array(losses), iterations_ics


def generate_target_and_reconstruct(
    model: Davis1985Simulation,
    target_seed: int = 42,
    reconstruction_seed: int = 1234,
    n_iterations: int = 200,
    learning_rate: float = 1e-1,
    loss_type: str = "chi2",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_iterations: bool = False,
    save_every: int = 1,
    init_scale: float = 0.1,
    noise_init: bool = False,
    noise_sigma: float = 0.001
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
        save_iterations: If True, save intermediate ICs during optimization
        save_every: Save ICs every N iterations
        init_scale: Scaling factor for initial random ICs (default 0.1)
        noise_init: If True, use pure Gaussian noise instead of cosmological ICs
        noise_sigma: Standard deviation for Gaussian noise (default 0.001)

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
    logger.info(f"Generating target ICs with seed {target_seed}")
    target_ics = model.generate_initial_conditions(seed=target_seed)

    # Run forward simulation
    logger.info(f"Running forward simulation with target seed {target_seed}")
    print(f"Running forward simulation with seed {target_seed}...")
    target_positions, _ = model.run_simulation(target_ics, rtol=rtol, atol=atol)

    # Paint densities at initial and final times
    target_density_init = model.paint_density(target_positions[0])
    target_density_final = model.paint_density(target_positions[-1])

    print(f"Target density (initial): min={target_density_init.min():.2e}, "
          f"mean={target_density_init.mean():.2e}, max={target_density_init.max():.2e}, "
          f"std={target_density_init.std():.2e}")
    print(f"Target density (final):   min={target_density_final.min():.2e}, "
          f"mean={target_density_final.mean():.2e}, max={target_density_final.max():.2e}, "
          f"std={target_density_final.std():.2e}")
    print()

    print("=" * 60)
    print("RECONSTRUCTING")
    print("=" * 60)

    # Reconstruct (optimize to match final density)
    reconstructed_ics, losses, iterations_ics = optimize_initial_conditions(
        target_density=target_density_final,
        model=model,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        seed=reconstruction_seed,
        loss_type=loss_type,
        rtol=rtol,
        atol=atol,
        save_iterations=save_iterations,
        save_every=save_every,
        init_scale=init_scale,
        noise_init=noise_init,
        noise_sigma=noise_sigma
    )

    # Run forward simulation with reconstructed ICs
    print("Running forward simulation with reconstructed ICs...")
    reconstructed_positions, _ = model.run_simulation(reconstructed_ics, rtol=rtol, atol=atol)
    reconstructed_density_init = model.paint_density(reconstructed_positions[0])
    reconstructed_density_final = model.paint_density(reconstructed_positions[-1])

    print(f"Reconstructed density (initial): min={reconstructed_density_init.min():.2e}, "
          f"mean={reconstructed_density_init.mean():.2e}, max={reconstructed_density_init.max():.2e}, "
          f"std={reconstructed_density_init.std():.2e}")
    print(f"Reconstructed density (final):   min={reconstructed_density_final.min():.2e}, "
          f"mean={reconstructed_density_final.mean():.2e}, max={reconstructed_density_final.max():.2e}, "
          f"std={reconstructed_density_final.std():.2e}")
    print()

    return {
        'target_ics': target_ics,
        'target_density_init': target_density_init,
        'target_density_final': target_density_final,
        'reconstructed_ics': reconstructed_ics,
        'reconstructed_density_init': reconstructed_density_init,
        'reconstructed_density_final': reconstructed_density_final,
        'losses': losses,
        'iterations_ics': iterations_ics,
    }


# =============================================================================
# 2D PROJECTION-BASED OPTIMIZATION
# =============================================================================


def projection_reconstruction_loss(
    initial_conditions: jnp.ndarray,
    target_projection: jnp.ndarray,
    model: Davis1985Simulation,
    loss_type: str = "chi2",
    projection_axis: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> float:
    """
    Loss function for 2D projection-based reconstruction.

    This function runs the forward simulation from initial conditions,
    paints the final density field, projects to 2D, and compares to the target.

    Args:
        initial_conditions: Current ICs being optimized (first arg for jax.grad)
        target_projection: Target 2D projected density to match
        model: Simulation model
        loss_type: Type of loss ("chi2" or "chi2_log")
        projection_axis: Axis to project along (0=x, 1=y, 2=z)
        rtol: Relative tolerance for ODE integrator
        atol: Absolute tolerance for ODE integrator

    Returns:
        Loss value (scalar)
    """
    from .projection import project_density_2d

    # Run forward simulation
    final_positions, _ = model.run_simulation(initial_conditions, rtol=rtol, atol=atol)

    # Paint final density field (last snapshot)
    predicted_density = model.paint_density(final_positions[-1])

    # Project to 2D
    predicted_projection = project_density_2d(predicted_density, axis=projection_axis)

    # Compute loss
    if loss_type == "chi2":
        return chi2(predicted_projection, target_projection)
    elif loss_type == "chi2_log":
        return chi2_log(predicted_projection, target_projection)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'chi2' or 'chi2_log'.")


def optimize_initial_conditions_2d(
    target_projection: jnp.ndarray,
    model: Davis1985Simulation,
    loss_fn: Optional[Callable] = None,
    n_iterations: int = 200,
    learning_rate: float = 1e-1,
    seed: int = 1234,
    print_every: int = 10,
    loss_type: str = "chi2",
    projection_axis: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_iterations: bool = False,
    save_every: int = 1,
    init_scale: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize initial conditions to match a target 2D projection.

    Uses Adam optimizer with JAX automatic differentiation to solve
    the inverse problem: find ICs that produce a final density whose
    2D projection matches the target projection.

    Args:
        target_projection: Target 2D projected density (shape: mesh_shape x mesh_shape)
        model: Davis1985Simulation instance
        loss_fn: Custom loss function(params, target, model) -> scalar.
                 If None, uses projection_reconstruction_loss with loss_type.
        n_iterations: Number of optimization iterations
        learning_rate: Adam optimizer learning rate
        seed: Random seed for initialization
        print_every: Print loss every N iterations
        loss_type: Type of loss ("chi2" or "chi2_log")
        projection_axis: Axis to project along (0=x, 1=y, 2=z)
        rtol: Relative tolerance for ODE integrator
        atol: Absolute tolerance for ODE integrator
        save_iterations: If True, save intermediate ICs during optimization
        save_every: Save ICs every N iterations (only if save_iterations=True)
        init_scale: Scaling factor for initial random ICs (default 0.1)

    Returns:
        Tuple of (optimized_ics, loss_history, iterations_ics)
        - optimized_ics: Optimized initial conditions
        - loss_history: Array of loss values during optimization
        - iterations_ics: List of ICs at saved iterations (empty if save_iterations=False)
    """
    print(f"Starting 2D projection optimization:")
    print(f"  Target projection shape: {target_projection.shape}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss type: {loss_type}")
    print(f"  Projection axis: {projection_axis}")
    print(f"  ODE tolerances: rtol={rtol}, atol={atol}")
    print()

    # Use default loss function if none provided
    if loss_fn is None:
        def loss_fn(ics, target, mdl):
            return projection_reconstruction_loss(
                ics, target, mdl,
                loss_type=loss_type,
                projection_axis=projection_axis,
                rtol=rtol,
                atol=atol
            )

    # Initialize from random ICs (scaled down for better convergence)
    logger.info(f"Generating initial conditions with seed {seed} (scale={init_scale})")
    print(f"Generating initial conditions with seed {seed} (scale={init_scale})...")
    params = init_scale * model.generate_initial_conditions(seed=seed)

    # Set up optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Create gradient function (differentiates w.r.t. first argument)
    grad_fn = jax.grad(loss_fn)

    # Optimization loop
    losses = []
    iterations_ics = []
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
        gradients = grad_fn(params, target_projection, model)

        # Update parameters
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        # Track progress - print first few iterations to show it's working
        should_print = (iteration < 3) or (iteration % print_every == 0) or (iteration == n_iterations - 1)

        if should_print:
            current_loss = loss_fn(params, target_projection, model)
            losses.append(float(current_loss))
            elapsed = time.time() - start_time
            print(f"{iteration:>10} | {current_loss:>12.6e} | {elapsed:>10.2f}", flush=True)

            # After first iteration, estimate remaining time
            if iteration == 0:
                avg_iter_time = iter_time
                est_total = avg_iter_time * n_iterations
                print(f"  -> First iteration took {iter_time:.1f}s. Estimated total: {est_total:.1f}s (~{est_total/60:.1f} min)", flush=True)

        # Save intermediate ICs if requested
        if save_iterations and (iteration % save_every == 0 or iteration == n_iterations - 1):
            iterations_ics.append((iteration, params.copy()))

    print()
    print("Optimization completed!")
    return params, jnp.array(losses), iterations_ics


def generate_target_and_reconstruct_2d(
    model: Davis1985Simulation,
    target_seed: int = 42,
    reconstruction_seed: int = 1234,
    n_iterations: int = 200,
    learning_rate: float = 1e-1,
    loss_type: str = "chi2",
    projection_axis: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_iterations: bool = False,
    save_every: int = 1,
    init_scale: float = 0.1
) -> dict:
    """
    Generate a target 2D projection and reconstruct it (for testing).

    This is a convenience function that:
    1. Generates ICs with target_seed
    2. Runs forward simulation to get target density
    3. Projects to 2D to get target projection
    4. Reconstructs ICs from target projection using different seed
    5. Returns all intermediate results for comparison

    Args:
        model: Davis1985Simulation instance
        target_seed: Seed for generating target ICs
        reconstruction_seed: Seed for initializing reconstruction
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimizer
        loss_type: Type of loss ("chi2" or "chi2_log")
        projection_axis: Axis to project along (0=x, 1=y, 2=z)
        rtol: Relative tolerance for ODE integrator
        atol: Absolute tolerance for ODE integrator
        save_iterations: If True, save intermediate ICs during optimization
        save_every: Save ICs every N iterations
        init_scale: Scaling factor for initial random ICs (default 0.1)

    Returns:
        Dictionary with:
        - 'target_ics': Original initial conditions
        - 'target_density_init': Target initial density field
        - 'target_density_final': Target final density field
        - 'target_projection': Target 2D projection
        - 'reconstructed_ics': Reconstructed initial conditions
        - 'reconstructed_density_init': Density from reconstructed ICs (initial)
        - 'reconstructed_density_final': Density from reconstructed ICs (final)
        - 'reconstructed_projection': Reconstructed 2D projection
        - 'losses': Loss history during optimization
        - 'iterations_ics': List of (iteration, ics) tuples
        - 'projection_axis': Axis used for projection
    """
    from .projection import project_density_2d

    print("=" * 60)
    print("GENERATING TARGET (2D PROJECTION)")
    print("=" * 60)

    # Generate target ICs
    logger.info(f"Generating target ICs with seed {target_seed}")
    target_ics = model.generate_initial_conditions(seed=target_seed)

    # Run forward simulation
    logger.info(f"Running forward simulation with target seed {target_seed}")
    print(f"Running forward simulation with seed {target_seed}...")
    target_positions, _ = model.run_simulation(target_ics, rtol=rtol, atol=atol)

    # Paint densities at initial and final times
    target_density_init = model.paint_density(target_positions[0])
    target_density_final = model.paint_density(target_positions[-1])

    # Create target 2D projection from final density
    target_projection = project_density_2d(target_density_final, axis=projection_axis)

    print(f"Target density (initial): min={target_density_init.min():.2e}, "
          f"mean={target_density_init.mean():.2e}, max={target_density_init.max():.2e}")
    print(f"Target density (final):   min={target_density_final.min():.2e}, "
          f"mean={target_density_final.mean():.2e}, max={target_density_final.max():.2e}")
    print(f"Target projection (2D):   min={target_projection.min():.2e}, "
          f"mean={target_projection.mean():.2e}, max={target_projection.max():.2e}")
    print()

    print("=" * 60)
    print("RECONSTRUCTING (2D PROJECTION)")
    print("=" * 60)

    # Reconstruct (optimize to match 2D projection)
    reconstructed_ics, losses, iterations_ics = optimize_initial_conditions_2d(
        target_projection=target_projection,
        model=model,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        seed=reconstruction_seed,
        loss_type=loss_type,
        projection_axis=projection_axis,
        rtol=rtol,
        atol=atol,
        save_iterations=save_iterations,
        save_every=save_every,
        init_scale=init_scale
    )

    # Run forward simulation with reconstructed ICs
    print("Running forward simulation with reconstructed ICs...")
    reconstructed_positions, _ = model.run_simulation(reconstructed_ics, rtol=rtol, atol=atol)
    reconstructed_density_init = model.paint_density(reconstructed_positions[0])
    reconstructed_density_final = model.paint_density(reconstructed_positions[-1])

    # Create reconstructed 2D projection
    reconstructed_projection = project_density_2d(reconstructed_density_final, axis=projection_axis)

    print(f"Reconstructed density (initial): min={reconstructed_density_init.min():.2e}, "
          f"mean={reconstructed_density_init.mean():.2e}, max={reconstructed_density_init.max():.2e}")
    print(f"Reconstructed density (final):   min={reconstructed_density_final.min():.2e}, "
          f"mean={reconstructed_density_final.mean():.2e}, max={reconstructed_density_final.max():.2e}")
    print(f"Reconstructed projection (2D):   min={reconstructed_projection.min():.2e}, "
          f"mean={reconstructed_projection.mean():.2e}, max={reconstructed_projection.max():.2e}")
    print()

    return {
        'target_ics': target_ics,
        'target_density_init': target_density_init,
        'target_density_final': target_density_final,
        'target_projection': target_projection,
        'reconstructed_ics': reconstructed_ics,
        'reconstructed_density_init': reconstructed_density_init,
        'reconstructed_density_final': reconstructed_density_final,
        'reconstructed_projection': reconstructed_projection,
        'losses': losses,
        'iterations_ics': iterations_ics,
        'projection_axis': projection_axis,
    }


def reconstruct_from_2d_target(
    target_projection: jnp.ndarray,
    model: Davis1985Simulation,
    reconstruction_seed: int = 1234,
    n_iterations: int = 200,
    learning_rate: float = 1e-1,
    loss_type: str = "chi2",
    projection_axis: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_iterations: bool = False,
    save_every: int = 1,
    init_scale: float = 0.1
) -> dict:
    """
    Reconstruct initial conditions from an external 2D target projection.

    This function takes a 2D target (e.g., from an extracted mask or observed image)
    and finds initial conditions that produce a matching 2D projection.

    Args:
        target_projection: External 2D target array to match
        model: Davis1985Simulation instance
        reconstruction_seed: Seed for initializing reconstruction
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimizer
        loss_type: Type of loss ("chi2" or "chi2_log")
        projection_axis: Axis to project along (0=x, 1=y, 2=z)
        rtol: Relative tolerance for ODE integrator
        atol: Absolute tolerance for ODE integrator
        save_iterations: If True, save intermediate ICs during optimization
        save_every: Save ICs every N iterations
        init_scale: Scaling factor for initial random ICs (default 0.1)

    Returns:
        Dictionary with:
        - 'target_projection': The input target 2D projection
        - 'reconstructed_ics': Reconstructed initial conditions
        - 'reconstructed_density_init': Density from reconstructed ICs (initial)
        - 'reconstructed_density_final': Density from reconstructed ICs (final)
        - 'reconstructed_projection': Reconstructed 2D projection
        - 'losses': Loss history during optimization
        - 'iterations_ics': List of (iteration, ics) tuples
        - 'projection_axis': Axis used for projection
    """
    from .projection import project_density_2d

    print("=" * 60)
    print("RECONSTRUCTING FROM EXTERNAL 2D TARGET")
    print("=" * 60)

    print(f"Target projection shape: {target_projection.shape}")
    print(f"Target projection:        min={target_projection.min():.2e}, "
          f"mean={target_projection.mean():.2e}, max={target_projection.max():.2e}")
    print()

    print("=" * 60)
    print("RECONSTRUCTING (2D PROJECTION)")
    print("=" * 60)

    # Reconstruct (optimize to match 2D projection)
    reconstructed_ics, losses, iterations_ics = optimize_initial_conditions_2d(
        target_projection=target_projection,
        model=model,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        seed=reconstruction_seed,
        loss_type=loss_type,
        projection_axis=projection_axis,
        rtol=rtol,
        atol=atol,
        save_iterations=save_iterations,
        save_every=save_every,
        init_scale=init_scale
    )

    # Run forward simulation with reconstructed ICs
    print("Running forward simulation with reconstructed ICs...")
    reconstructed_positions, _ = model.run_simulation(reconstructed_ics, rtol=rtol, atol=atol)
    reconstructed_density_init = model.paint_density(reconstructed_positions[0])
    reconstructed_density_final = model.paint_density(reconstructed_positions[-1])

    # Create reconstructed 2D projection
    reconstructed_projection = project_density_2d(reconstructed_density_final, axis=projection_axis)

    print(f"Reconstructed density (initial): min={reconstructed_density_init.min():.2e}, "
          f"mean={reconstructed_density_init.mean():.2e}, max={reconstructed_density_init.max():.2e}")
    print(f"Reconstructed density (final):   min={reconstructed_density_final.min():.2e}, "
          f"mean={reconstructed_density_final.mean():.2e}, max={reconstructed_density_final.max():.2e}")
    print(f"Reconstructed projection (2D):   min={reconstructed_projection.min():.2e}, "
          f"mean={reconstructed_projection.mean():.2e}, max={reconstructed_projection.max():.2e}")
    print()

    return {
        'target_projection': target_projection,
        'reconstructed_ics': reconstructed_ics,
        'reconstructed_density_init': reconstructed_density_init,
        'reconstructed_density_final': reconstructed_density_final,
        'reconstructed_projection': reconstructed_projection,
        'losses': losses,
        'iterations_ics': iterations_ics,
        'projection_axis': projection_axis,
    }
