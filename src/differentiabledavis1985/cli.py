"""CLI for DifferentiableDavis1985 - Reproducing Davis et al. (1985) simulations."""

import typer

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


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
