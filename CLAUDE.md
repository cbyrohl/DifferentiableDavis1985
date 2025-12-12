# Development Notes

## Python Environment

Use `uv run` to execute Python scripts and commands within the project environment:

```bash
uv run python script.py
uv run davis1985 <command>
```

## CLI Tool

The project provides a `davis1985` CLI tool built with Typer.

Available commands:
- `uv run davis1985 info` - Show project information
- `uv run davis1985 plot` - Run simulation and generate density field plots
- `uv run davis1985 extract-mask [image]` - Extract particle mask from scatter plot image (default: paper_material/fig1_lowerleft_original.png)

## Configuration Files

Simulation parameters can be stored in YAML config files in the `configs/` directory.

Available configs:
- `configs/default.yaml` - Default parameters
- `configs/validation.yaml` - Simple validation case (64^3 particles, a=0.01→0.33, z=99→2)
- `configs/davis1985.yaml` - Model EdS1 from Davis et al. (1985) Figure 1 (Einstein-de Sitter, Ω=1)
- `configs/davis1985_open.yaml` - Model O1 from Davis et al. (1985) Figure 1 (Open universe, Ω evolves)

Usage:
```bash
# Use config file
uv run davis1985 plot -c configs/davis1985.yaml

# Override config values with CLI arguments
uv run davis1985 plot -c configs/davis1985.yaml -n 16 --omega-m 0.3

# Use defaults (no config file)
uv run davis1985 plot -n 32
```

Config parameters:
- `n_particles`: Number of particles per dimension (total = n_particles^3)
- `omega_m`: Matter density parameter (Ω_m)
- `a_init`: Initial scale factor (a = 1/(1+z))
- `a_final`: Final scale factor
- `box_size`: Box size in comoving Mpc/h
- `mesh_shape`: Mesh cells per dimension (null = auto-calculate)
- `seed`: Random seed for initial conditions

**Specific config examples:**
- Validation: 64^3 particles, Ω=0.3, L=100 Mpc/h, a=0.01→0.33 (z=99→2), auto mesh
- EdS1 model: 32^3 particles, Ω=1.0, L=32.5 h^{-2} Mpc, a=1.0→2.4, 64^3 mesh
- O1 model: 32^3 particles, Ω=0.2, L=162.5 h^{-2} Mpc, a=1.0→3.2, 64^3 mesh
