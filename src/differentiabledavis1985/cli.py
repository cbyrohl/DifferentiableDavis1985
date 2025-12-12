"""CLI for DifferentiableDavis1985 - Reproducing Davis et al. (1985) simulations."""

import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

from .simulation import run_nbody_simulation
from .plotting import plot_density_slice_from_cube, plot_density_comparison
from .config import SimulationConfig
from .forward_model import Davis1985Simulation
from .optimization import generate_target_and_reconstruct

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
):
    """Run simulation and plot density field slices."""
    import matplotlib.pyplot as plt

    # Load config from file or use defaults
    if config is not None:
        sim_config = SimulationConfig.from_yaml(config)
        typer.echo(f"Loaded config from: {config}")
    else:
        sim_config = SimulationConfig()

    # Override with command-line arguments
    if n_particles is not None:
        sim_config.n_particles = n_particles
    if mesh_shape is not None:
        sim_config.mesh_shape = mesh_shape
    if box_size is not None:
        sim_config.box_size = box_size
    if omega_m is not None:
        sim_config.omega_m = omega_m
    if seed is not None:
        sim_config.seed = seed

    # Handle redshift/scale factor conversion
    if z_init is not None:
        sim_config.a_init = 1.0 / (1.0 + z_init)
    if z_final is not None:
        sim_config.a_final = 1.0 / (1.0 + z_final)

    # Auto-calculate mesh_shape if still None
    if mesh_shape is None and config is None:
        sim_config.mesh_shape = sim_config.n_particles // 2

    typer.echo(f"\n{sim_config}")
    typer.echo(f"\nUsing mesh: {sim_config.mesh_shape}^3 = {sim_config.mesh_shape**3} cells "
               f"({sim_config.n_particles**3 // sim_config.mesh_shape**3}x fewer than particles)\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    typer.echo(f"Running simulation and generating plots...")

    results = run_nbody_simulation(
        n_particles=sim_config.n_particles,
        box_size=sim_config.box_size,
        mesh_shape=sim_config.mesh_shape,
        z_init=sim_config.z_init,
        z_final=sim_config.z_final,
        omega_m=sim_config.omega_m,
        seed=sim_config.seed,
    )

    typer.echo("\nPlotting initial density field...")
    fig_init, _ = plot_density_slice_from_cube(
        results['density_init'],
        results['box_size'],
    )
    fig_init.suptitle(f"Initial Density Field (z={sim_config.z_init:.1f}, a={sim_config.a_init:.3f})")
    init_path = output_path / "density_initial.png"
    fig_init.savefig(init_path, dpi=150, bbox_inches='tight')
    typer.echo(f"  Saved to: {init_path}")
    plt.close(fig_init)

    typer.echo("\nPlotting final density field...")
    fig_final, _ = plot_density_slice_from_cube(
        results['density_final'],
        results['box_size'],
    )
    fig_final.suptitle(f"Final Density Field (z={sim_config.z_final:.1f}, a={sim_config.a_final:.3f})")
    final_path = output_path / "density_final.png"
    fig_final.savefig(final_path, dpi=150, bbox_inches='tight')
    typer.echo(f"  Saved to: {final_path}")
    plt.close(fig_final)

    typer.echo(f"\nDone! Plots saved to {output_path}/")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
