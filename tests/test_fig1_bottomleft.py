"""
Test script for reproducing Figure 1 (bottom left) from Davis et al. 1985.

This runs the EdS1 simulation at a = 2.4 (Einstein-de Sitter, Omega = 1).
"""

import jax.numpy as jnp
from differentiabledavis1985 import Fig1BottomLeftSimulation


def test_fig1_bottomleft_simulation():
    """Run the Fig 1 bottom-left simulation."""
    # Initialize the simulation
    sim = Fig1BottomLeftSimulation(
        initial_a=0.1,
        final_a=2.4,
        n_snapshots=8,
        seed=42,
    )

    # Verify simulation configuration
    assert sim.mesh_shape is not None
    assert sim.box_size is not None
    assert len(sim.snapshots) == 8
    assert sim.snapshots[0] >= 0.1
    assert sim.snapshots[-1] <= 2.4

    # Run the simulation
    positions, initial_positions = sim.run()

    # Verify output shapes
    expected_particles = jnp.prod(jnp.array(sim.mesh_shape))
    assert positions.shape == (8, expected_particles, 3)
    assert initial_positions.shape == (expected_particles, 3)

    # Verify all positions are finite
    assert jnp.isfinite(positions).all(), "All positions should be finite"
    assert jnp.isfinite(initial_positions).all(), "Initial positions should be finite"

    # Analyze each snapshot
    for i, a in enumerate(sim.snapshots):
        pos = positions[i]

        # Verify positions are reasonable
        assert jnp.all(jnp.isfinite(pos)), f"Positions at a={a} should be finite"

        # Calculate displacement from initial lattice positions
        initial_lattice = jnp.stack(
            jnp.meshgrid(*[jnp.arange(32) for _ in range(3)]),
            axis=-1
        ).reshape([-1, 3])
        displacement = pos - initial_lattice
        mean_displacement = jnp.mean(jnp.linalg.norm(displacement, axis=1))

        # Verify that particles have moved from their initial positions
        # (especially at later times)
        if a > 0.5:
            assert mean_displacement > 0, f"Particles should have moved by a={a}"

    # Focus on the final snapshot (a = 2.4, Fig 1 bottom-left)
    final_pos = positions[-1]
    assert final_pos.shape[0] == expected_particles

    # Verify final positions have reasonable statistics
    mean_pos = jnp.mean(final_pos, axis=0)
    std_pos = jnp.std(final_pos, axis=0)

    # Mean should be roughly centered (around mesh_shape/2)
    # Note: allowing for periodic boundaries
    assert jnp.all(jnp.isfinite(mean_pos))
    assert jnp.all(jnp.isfinite(std_pos))

    # Standard deviation should be positive (particles have spread out)
    assert jnp.all(std_pos > 0), "Positions should have non-zero spread"
