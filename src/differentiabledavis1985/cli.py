"""CLI for DifferentiableDavis1985 - Reproducing Davis et al. (1985) simulations."""

import sys
import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional
from loguru import logger

from .simulation import run_nbody_simulation
from .plotting import plot_density_slice_from_cube, plot_density_comparison, plot_overdensity_comparison
from .config import SimulationConfig
from .forward_model import Davis1985Simulation
from .optimization import generate_target_and_reconstruct
from .extract_mask import extract_particle_mask


def configure_logging(verbose: bool = True):
    """Configure loguru logging.

    Args:
        verbose: If True, use DEBUG level. Otherwise use INFO level.
    """
    logger.remove()  # Remove default handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.debug(f"Logging configured with level={level}")


app = typer.Typer(
    name="davis1985",
    help="Differentiable N-body simulations reproducing Davis et al. (1985)",
    add_completion=False,
)


@app.command()
def info():
    """Display information about the Davis et al. (1985) simulation."""
    typer.echo("DifferentiableDavis1985")
    typer.echo("=" * 50)
    typer.echo("Reproducing: Davis, M., Efstathiou, G., Frenk, C. S., & White, S. D. M. (1985)")
    typer.echo("ApJ, 292, 371-394")
    typer.echo("")
    typer.echo("Original simulation parameters:")
    typer.echo("  - 32,768 particles")
    typer.echo("  - CDM-dominated universe")
    typer.echo("  - Demonstrated formation of large-scale structure")


@app.command()
def plot(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to YAML config file")] = None,
    n_particles: Annotated[Optional[int], typer.Option("--particles", "-n", help="Particles per dimension (overrides config)")] = None,
    mesh_shape: Annotated[Optional[int], typer.Option("--mesh", "-m", help="Mesh cells per dimension (overrides config)")] = None,
    box_size: Annotated[Optional[float], typer.Option("--box-size", "-L", help="Box size in Mpc/h (overrides config)")] = None,
    omega_m: Annotated[Optional[float], typer.Option("--omega-m", help="Matter density parameter (overrides config)")] = None,
    z_init: Annotated[Optional[float], typer.Option("--z-init", help="Initial redshift (overrides config)")] = None,
    z_final: Annotated[Optional[float], typer.Option("--z-final", help="Final redshift (overrides config)")] = None,
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed (overrides config)")] = None,
    n_steps_min: Annotated[Optional[int], typer.Option("--n-steps-min", help="Min output snapshots for ODE integrator (overrides config)")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging")] = True,
):
    """Run simulation and plot density field slices."""
    import matplotlib.pyplot as plt

    configure_logging(verbose=verbose)
    logger.info("Starting plot command")

    # Load config from file or use defaults
    if config is not None:
        logger.info(f"Loading config from: {config}")
        sim_config = SimulationConfig.from_yaml(config)
        logger.debug(f"Config loaded: {sim_config}")
    else:
        logger.info("Using default config")
        sim_config = SimulationConfig()

    # Override with command-line arguments
    if n_particles is not None:
        logger.debug(f"Overriding n_particles: {sim_config.n_particles} -> {n_particles}")
        sim_config.n_particles = n_particles
    if mesh_shape is not None:
        logger.debug(f"Overriding mesh_shape: {sim_config.mesh_shape} -> {mesh_shape}")
        sim_config.mesh_shape = mesh_shape
    if box_size is not None:
        logger.debug(f"Overriding box_size: {sim_config.box_size} -> {box_size}")
        sim_config.box_size = box_size
    if omega_m is not None:
        logger.debug(f"Overriding omega_m: {sim_config.omega_m} -> {omega_m}")
        sim_config.omega_m = omega_m
    if seed is not None:
        logger.debug(f"Overriding seed: {sim_config.seed} -> {seed}")
        sim_config.seed = seed
    if n_steps_min is not None:
        logger.debug(f"Overriding n_steps_min: {sim_config.n_steps_min} -> {n_steps_min}")
        sim_config.n_steps_min = n_steps_min

    # Handle redshift/scale factor conversion
    if z_init is not None:
        new_a_init = 1.0 / (1.0 + z_init)
        logger.debug(f"Overriding a_init: {sim_config.a_init} -> {new_a_init} (z={z_init})")
        sim_config.a_init = new_a_init
    if z_final is not None:
        new_a_final = 1.0 / (1.0 + z_final)
        logger.debug(f"Overriding a_final: {sim_config.a_final} -> {new_a_final} (z={z_final})")
        sim_config.a_final = new_a_final

    # Auto-calculate mesh_shape if still None
    if mesh_shape is None and config is None:
        sim_config.mesh_shape = sim_config.n_particles // 2
        logger.debug(f"Auto-calculated mesh_shape: {sim_config.mesh_shape}")

    logger.info(f"Final config: {sim_config}")
    logger.info(f"Using mesh: {sim_config.mesh_shape}^3 = {sim_config.mesh_shape**3} cells "
                f"({sim_config.n_particles**3 // sim_config.mesh_shape**3}x fewer than particles)")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Output directory: {output_path}")

    logger.info("Running simulation...")

    results = run_nbody_simulation(
        n_particles=sim_config.n_particles,
        box_size=sim_config.box_size,
        mesh_shape=sim_config.mesh_shape,
        z_init=sim_config.z_init,
        z_final=sim_config.z_final,
        omega_m=sim_config.omega_m,
        seed=sim_config.seed,
        n_steps_min=sim_config.n_steps_min,
    )

    # Determine filename suffix
    suffix = f"_{sim_config.suffix}" if sim_config.suffix else ""
    logger.debug(f"Filename suffix: '{suffix}'")

    logger.info("Plotting overdensity field comparison...")
    fig, _ = plot_overdensity_comparison(
        results['density_init'],
        results['density_final'],
        results['box_size'],
        a_init=sim_config.a_init,
        a_final=sim_config.a_final,
    )
    comparison_path = output_path / f"overdensity{suffix}.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to: {comparison_path}")
    plt.close(fig)

    logger.info(f"Done! Plots saved to {output_path}/")


@app.command()
def reconstruct(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to YAML config file")] = None,
    mesh_shape: Annotated[Optional[int], typer.Option("--mesh", "-m", help="Mesh cells per dimension (overrides config)")] = None,
    box_size: Annotated[Optional[float], typer.Option("--box-size", "-L", help="Box size in Mpc/h (overrides config)")] = None,
    z_init: Annotated[Optional[float], typer.Option("--z-init", help="Initial redshift (overrides config)")] = None,
    z_final: Annotated[Optional[float], typer.Option("--z-final", help="Final redshift (overrides config)")] = None,
    omega_c: Annotated[float, typer.Option("--omega-c", help="CDM density parameter")] = 0.25,
    sigma8: Annotated[float, typer.Option("--sigma8", help="Matter fluctuation amplitude")] = 0.8,
    target_seed: Annotated[int, typer.Option("--target-seed", help="Seed for generating target")] = 42,
    reconstruction_seed: Annotated[int, typer.Option("--recon-seed", help="Seed for reconstruction init")] = 1234,
    n_iterations: Annotated[int, typer.Option("--iterations", "-i", help="Number of optimization iterations")] = 100,
    learning_rate: Annotated[float, typer.Option("--lr", help="Learning rate for Adam optimizer")] = 1e-1,
    loss_type: Annotated[str, typer.Option("--loss-type", help="Loss function (chi2 or chi2_log)")] = "chi2",
    rtol: Annotated[float, typer.Option("--rtol", help="ODE relative tolerance (lower=faster, default 1e-4)")] = 1e-4,
    atol: Annotated[float, typer.Option("--atol", help="ODE absolute tolerance (lower=faster, default 1e-4)")] = 1e-4,
    n_steps_min: Annotated[Optional[int], typer.Option("--n-steps-min", help="Min output snapshots for ODE integrator (overrides config)")] = None,
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output/reconstruction",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging")] = True,
):
    """Reconstruct initial conditions from target final density field.

    This command demonstrates the differentiable capabilities by:
    1. Generating target ICs and running forward simulation
    2. Reconstructing ICs from the final density using gradient descent
    3. Comparing target vs reconstructed fields
    """
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import numpy as np

    configure_logging(verbose=verbose)

    logger.info("=" * 70)
    logger.info("DENSITY FIELD RECONSTRUCTION")
    logger.info("=" * 70)

    # Load config from file or use defaults
    if config is not None:
        sim_config = SimulationConfig.from_yaml(config)
        logger.info(f"Loaded config from: {config}")
    else:
        sim_config = SimulationConfig()
        logger.info("Using default configuration")

    # Override with command-line arguments
    if mesh_shape is not None:
        sim_config.mesh_shape = mesh_shape
    if box_size is not None:
        sim_config.box_size = box_size
    if z_init is not None:
        sim_config.a_init = 1.0 / (1.0 + z_init)
    if z_final is not None:
        sim_config.a_final = 1.0 / (1.0 + z_final)

    # Auto-calculate mesh_shape if still None and no config
    if sim_config.mesh_shape is None and config is None:
        sim_config.mesh_shape = 16  # Default for reconstruction

    # Determine n_steps_min
    if n_steps_min is None:
        n_steps_min = getattr(sim_config, 'n_steps_min', 2)

    # Set up model
    mesh_shape_tuple = (sim_config.mesh_shape, sim_config.mesh_shape, sim_config.mesh_shape)
    a_init = sim_config.a_init
    a_final = sim_config.a_final
    snapshots = jnp.linspace(a_init, a_final, n_steps_min)

    logger.info("Simulation parameters:")
    logger.info(f"  Mesh: {sim_config.mesh_shape}^3 = {sim_config.mesh_shape**3} cells")
    logger.info(f"  Box size: {sim_config.box_size} Mpc/h")
    logger.info(f"  Redshift: {sim_config.z_init:.1f} -> {sim_config.z_final:.1f} (a: {a_init:.4f} -> {a_final:.4f})")
    logger.info(f"  n_steps_min: {n_steps_min}")
    logger.info(f"  Omega_c: {omega_c}, sigma8: {sigma8}")
    logger.info("Reconstruction parameters:")
    logger.info(f"  Iterations: {n_iterations}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Loss type: {loss_type}")

    model = Davis1985Simulation(
        mesh_shape=mesh_shape_tuple,
        box_size=sim_config.box_size,
        snapshots=snapshots,
        omega_c=omega_c,
        sigma8=sigma8,
    )

    # Run reconstruction
    logger.info("Starting reconstruction...")
    results = generate_target_and_reconstruct(
        model=model,
        target_seed=target_seed,
        reconstruction_seed=reconstruction_seed,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        loss_type=loss_type,
        rtol=rtol,
        atol=atol
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Output directory: {output_path}")

    # Save results
    logger.info("Saving results...")
    np.save(output_path / "target_ics.npy", np.array(results['target_ics']))
    np.save(output_path / "target_density.npy", np.array(results['target_density']))
    np.save(output_path / "reconstructed_ics.npy", np.array(results['reconstructed_ics']))
    np.save(output_path / "reconstructed_density.npy", np.array(results['reconstructed_density']))
    np.save(output_path / "losses.npy", np.array(results['losses']))
    logger.debug(f"Saved arrays to {output_path}/")

    # Plot loss convergence
    logger.info("Plotting loss convergence...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results['losses'], marker='o')
    ax.set_xlabel("Iteration (x10)")
    ax.set_ylabel("Loss")
    ax.set_title("Reconstruction Loss Convergence")
    ax.grid(True, alpha=0.3)
    loss_path = output_path / "loss_convergence.png"
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    logger.debug(f"Saved to: {loss_path}")
    plt.close(fig)

    # Plot overdensity comparison: Target
    logger.info("Plotting target overdensity evolution...")
    fig, _ = plot_overdensity_comparison(
        results['target_ics'],
        results['target_density'],
        sim_config.box_size,
        a_init=a_init,
        a_final=a_final,
    )
    fig.suptitle("Target Overdensity Evolution", fontsize=14, y=0.98)
    target_path = output_path / "overdensity_target.png"
    fig.savefig(target_path, dpi=150, bbox_inches='tight')
    logger.debug(f"Saved to: {target_path}")
    plt.close(fig)

    # Plot overdensity comparison: Reconstructed
    logger.info("Plotting reconstructed overdensity evolution...")
    fig, _ = plot_overdensity_comparison(
        results['reconstructed_ics'],
        results['reconstructed_density'],
        sim_config.box_size,
        a_init=a_init,
        a_final=a_final,
    )
    fig.suptitle("Reconstructed Overdensity Evolution", fontsize=14, y=0.98)
    recon_path = output_path / "overdensity_reconstructed.png"
    fig.savefig(recon_path, dpi=150, bbox_inches='tight')
    logger.debug(f"Saved to: {recon_path}")
    plt.close(fig)

    # Print final statistics
    logger.info("=" * 70)
    logger.info("RECONSTRUCTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final loss: {results['losses'][-1]:.6e}")
    logger.info(f"Results saved to: {output_path}/")


@app.command("demonstration-plots")
def demonstration_plots(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML config file")] = "configs/validation.yaml",
    n_snapshots: Annotated[int, typer.Option("--snapshots", "-n", help="Number of snapshots")] = 20,
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output/demonstration",
    fps: Annotated[int, typer.Option("--fps", help="Frames per second for GIF")] = 5,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = True,
):
    """Create demonstration plots showing time evolution.

    Generates an animated GIF showing the evolution of structure from z=100 to z=0.
    """
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import numpy as np
    import imageio
    from io import BytesIO

    configure_logging(verbose=verbose)

    logger.info("=" * 70)
    logger.info("DEMONSTRATION PLOTS: TIME EVOLUTION")
    logger.info("=" * 70)

    # Load config
    logger.info(f"Loading config from: {config}")
    sim_config = SimulationConfig.from_yaml(config)

    # Override for demonstration: z=100 to z=0
    sim_config.a_init = 1.0 / (1.0 + 100.0)  # z=100
    sim_config.a_final = 1.0  # z=0
    sim_config.n_steps_min = n_snapshots

    # Auto-calculate mesh_shape if needed
    if sim_config.mesh_shape is None:
        sim_config.mesh_shape = sim_config.n_particles // 2
        logger.debug(f"Auto-calculated mesh_shape: {sim_config.mesh_shape}")

    logger.info(f"Simulation parameters:")
    logger.info(f"  Particles: {sim_config.n_particles}^3 = {sim_config.n_particles**3:,}")
    logger.info(f"  Mesh: {sim_config.mesh_shape}^3 = {sim_config.mesh_shape**3:,}")
    logger.info(f"  Box size: {sim_config.box_size} Mpc/h")
    logger.info(f"  Redshift: 100 -> 0")
    logger.info(f"  Snapshots: {n_snapshots}")
    logger.info(f"  FPS for GIF: {fps}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Run simulation
    logger.info("Running N-body simulation...")
    logger.info("(This will take a few minutes for time evolution...)")

    results = run_nbody_simulation(
        n_particles=sim_config.n_particles,
        box_size=sim_config.box_size,
        mesh_shape=sim_config.mesh_shape,
        z_init=100.0,
        z_final=0.0,
        omega_m=sim_config.omega_m,
        seed=sim_config.seed,
        n_steps_min=n_snapshots,
    )

    logger.info(f"Simulation complete! Generated {len(results['densities'])} snapshots")

    # Generate frames for GIF
    logger.info("Generating animation frames...")
    frames = []
    scale_factors = results['scale_factors']

    for i, (density, a) in enumerate(zip(results['densities'], scale_factors)):
        z = 1.0/a - 1.0

        # Convert to overdensity
        from .plotting import density_to_overdensity
        delta = density_to_overdensity(density)

        # Get middle slice
        slc_idx = density.shape[2] // 2
        slc = delta[:, :, slc_idx]

        # Create frame
        fig, ax = plt.subplots(figsize=(8, 8))

        vmax = np.percentile(np.abs(slc), 99)
        vmin = -vmax

        im = ax.imshow(
            slc.T,
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            extent=[0, sim_config.box_size, 0, sim_config.box_size],
            origin="lower",
        )

        ax.set_xlabel("cMpc/h", fontsize=12)
        ax.set_ylabel("cMpc/h", fontsize=12)
        ax.set_title(f"Overdensity Field Evolution\nz = {z:.1f}, a = {a:.3f}", fontsize=14, pad=10)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"overdensity $\delta$", fontsize=12)

        # Save frame to bytes (without bbox_inches='tight' for consistent sizing)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

        if (i + 1) % 5 == 0:
            logger.debug(f"Generated frame {i+1}/{n_snapshots}")

    # Save as GIF
    gif_path = output_path / "time_evolution.gif"
    logger.info(f"Saving GIF animation to: {gif_path}")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)

    logger.info("=" * 70)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"GIF saved to: {gif_path}")
    logger.info(f"  Snapshots: {n_snapshots}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Duration: {n_snapshots / fps:.1f} seconds")


@app.command("extract-mask")
def extract_mask(
    image: Annotated[str, typer.Argument(help="Path to input image (scatter plot)")] = "paper_material/fig1_lowerleft_original.png",
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    darkness_threshold: Annotated[float, typer.Option("--threshold", "-t", help="Min darkness %% (0-100) for particle detection. Lower=stricter, higher=more lenient")] = 2.0,
    resolution: Annotated[Optional[int], typer.Option("--resolution", "-r", help="Target resolution (e.g., 64, 128, 256)")] = None,
    no_square: Annotated[bool, typer.Option("--no-square", help="Don't enforce square shape")] = False,
    no_frame_removal: Annotated[bool, typer.Option("--no-frame-removal", help="Skip automatic frame removal")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = True,
):
    """Extract particle mask from a scatter plot image.

    Reads an image containing a particle scatter plot, removes the frame/border,
    and extracts a binary mask where particles=1 and background=0.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    configure_logging(verbose=verbose)

    logger.info(f"Extracting mask from: {image}")

    result = extract_particle_mask(
        image_path=image,
        darkness_threshold=darkness_threshold,
        remove_frame=not no_frame_removal,
        square=not no_square,
        resolution=resolution,
    )

    if result is None:
        logger.error("Failed to extract mask")
        raise typer.Exit(1)

    mask, cropped_img = result

    # Compute statistics
    total_pixels = mask.size
    particle_pixels = np.sum(mask)
    fraction = particle_pixels / total_pixels

    logger.info("=" * 50)
    logger.info("MASK STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Shape: {mask.shape[0]} x {mask.shape[1]} pixels")
    logger.info(f"Total pixels: {total_pixels:,}")
    logger.info(f"Particle pixels: {particle_pixels:,}")
    logger.info(f"Background pixels: {total_pixels - particle_pixels:,}")
    logger.info(f"Particle fraction: {fraction:.4f} ({fraction*100:.2f}%)")
    logger.info("=" * 50)

    output_path = Path(output)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save debug comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title("Original Image")
    axes[0].imshow(Image.open(image), cmap='gray')
    axes[0].axis('off')

    axes[1].set_title("Cropped (Frame Removed)")
    axes[1].imshow(cropped_img, cmap='gray')
    axes[1].axis('off')

    axes[2].set_title("Extracted Particle Mask")
    axes[2].imshow(mask, cmap='gray', interpolation='none')
    axes[2].axis('off')

    plt.tight_layout()
    debug_path = output_path / "mask_extraction_debug.png"
    fig.savefig(debug_path, dpi=150)
    logger.info(f"Debug comparison saved to: {debug_path}")
    plt.close(fig)

    # Save mask image
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mask, cmap='gray', interpolation='none')
    ax.set_title(f"Particle Mask ({mask.shape[0]}x{mask.shape[1]})")
    ax.axis('off')
    plt.tight_layout()
    mask_img_path = output_path / "particle_mask.png"
    fig2.savefig(mask_img_path, dpi=150, bbox_inches='tight')
    logger.info(f"Mask image saved to: {mask_img_path}")
    plt.close(fig2)

    # Save numpy array
    mask_npy_path = output_path / "particle_mask.npy"
    np.save(mask_npy_path, mask)
    logger.info(f"Mask numpy array saved to: {mask_npy_path}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
