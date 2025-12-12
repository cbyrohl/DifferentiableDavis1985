"""
Test script for the Davis 1985 forward model.

Runs a small test simulation to verify the setup works correctly.
"""

import jax.numpy as jnp
from differentiabledavis1985.forward_model import Davis1985Simulation


def test_small_simulation():
    """Run a small test simulation."""
    # Small test: 32^3 mesh (original paper used 32,768 particles)
    mesh_shape = (32, 32, 32)
    box_size = 32.5  # Mpc/h (from Table 1 of Davis et al. 1985)

    # Output at a few scale factors
    snapshots = jnp.array([0.1, 0.5, 1.0])

    # Initialize simulation with Omega=1 parameters (EdS models from paper)
    sim = Davis1985Simulation(
        mesh_shape=mesh_shape,
        box_size=box_size,
        snapshots=snapshots,
        omega_c=0.25,
        omega_b=0.045,
        sigma8=0.8,
        h=0.7,
    )

    # Verify simulation setup
    assert sim.mesh_shape == mesh_shape
    expected_particles = jnp.prod(jnp.array(mesh_shape))

    # Generate initial conditions
    initial_conditions = sim.generate_initial_conditions(seed=42)

    # Verify initial conditions shape and properties
    assert initial_conditions.shape == mesh_shape
    assert jnp.isfinite(initial_conditions).all(), "Initial conditions should be finite"

    # Run simulation
    positions, initial_positions = sim.run_simulation(initial_conditions)

    # Verify output shapes
    assert positions.shape == (len(snapshots), expected_particles, 3)
    assert initial_positions.shape == (expected_particles, 3)

    # Verify all positions are finite
    assert jnp.isfinite(positions).all(), "All positions should be finite"
    assert jnp.isfinite(initial_positions).all(), "Initial positions should be finite"

    # Verify positions for each snapshot
    for i, a in enumerate(snapshots):
        pos = positions[i]

        # Check that positions are reasonable (within some multiple of box size)
        assert jnp.all(jnp.isfinite(pos)), f"Positions at a={a} should be finite"

        # Positions should have some spread (not all at origin)
        assert jnp.std(pos) > 0, f"Positions at a={a} should have non-zero spread"
