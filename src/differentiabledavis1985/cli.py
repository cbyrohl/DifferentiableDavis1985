"""CLI for DifferentiableDavis1985 - Reproducing Davis et al. (1985) simulations."""

import typer
from typing_extensions import Annotated
from pathlib import Path

from .simulation import run_nbody_simulation
from .plotting import plot_density_slice_from_cube

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
    n_particles: Annotated[int, typer.Option("--particles", "-n", help="Particles per dimension")] = 32,
    mesh_shape: Annotated[int, typer.Option("--mesh", "-m", help="Mesh cells per dimension")] = 64,
    box_size: Annotated[float, typer.Option("--box-size", "-L", help="Box size in Mpc/h")] = 100.0,
    z_init: Annotated[float, typer.Option("--z-init", help="Initial redshift")] = 99.0,
    z_final: Annotated[float, typer.Option("--z-final", help="Final redshift")] = 0.0,
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
):
    """Run simulation and plot density field slices."""
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    typer.echo(f"Running simulation and generating plots...")

    results = run_nbody_simulation(
        n_particles=n_particles,
        box_size=box_size,
        mesh_shape=mesh_shape,
        z_init=z_init,
        z_final=z_final,
        seed=seed,
    )

    typer.echo("\nPlotting initial density field...")
    fig_init, _ = plot_density_slice_from_cube(
        results['density_init'],
        results['box_size'],
    )
    fig_init.suptitle(f"Initial Density Field (z={z_init})")
    init_path = output_path / "density_initial.png"
    fig_init.savefig(init_path, dpi=150, bbox_inches='tight')
    typer.echo(f"  Saved to: {init_path}")
    plt.close(fig_init)

    typer.echo("\nPlotting final density field...")
    fig_final, _ = plot_density_slice_from_cube(
        results['density_final'],
        results['box_size'],
    )
    fig_final.suptitle(f"Final Density Field (z={z_final})")
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
