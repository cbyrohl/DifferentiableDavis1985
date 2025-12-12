# DifferentiableDavis1985

Reproducing particle distributions from Davis et al. (1985) "The Evolution of Large-Scale Structure in a Universe Dominated by Cold Dark Matter" using differentiable forward simulations.

## Goal

Reproduce the N-body simulation results from the seminal CDM paper (see `paper_material/paper.pdf`) using modern differentiable simulation techniques powered by JAX and jaxpm.

## Original Paper

Davis, M., Efstathiou, G., Frenk, C. S., & White, S. D. M. (1985)
ApJ, 292, 371-394

The paper presented simulations of gravitational clustering in cold dark matter dominated universes with 32,768 particles, demonstrating the formation of large-scale structure, filaments, and voids.

## Current Status

✅ **Fig 1 (bottom left) simulation implemented** - EdS1 model at a = 2.4

### Implemented Features

- Einstein-de Sitter (Ω = 1) forward model
- 32³ = 32,768 particles matching original paper
- Box size: 32.5 h⁻¹ Mpc
- Evolution from a = 0.1 to a = 2.4
- Differentiable gravitational N-body solver via jaxpm

## Quick Start

```bash
# Install dependencies using uv
uv sync

# Run the Fig 1 bottom-left simulation
uv run python test_fig1_bottomleft.py
```

## Usage

```python
from differentiabledavis1985 import Fig1BottomLeftSimulation

# Initialize Fig1 bottom-left (EdS1 at a=2.4)
sim = Fig1BottomLeftSimulation(
    initial_a=0.1,
    final_a=2.4,
    n_snapshots=8,
    seed=42
)

# Run simulation
positions, initial_positions = sim.run()

# positions.shape = (n_snapshots, 32768, 3)
```

## Physics Engine

Built on [jaxpm](https://github.com/DifferentiableUniverseInitiative/jaxpm) - a differentiable particle-mesh N-body solver in JAX.
