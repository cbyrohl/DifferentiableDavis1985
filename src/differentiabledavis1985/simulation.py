"""Simple N-body simulation runner using jaxpm."""

import jax
import jax.numpy as jnp
import jaxpm
from jaxpm.painting_utils import scatter


def run_nbody_simulation(
    n_particles=32,
    box_size=100.0,
    mesh_shape=64,
    z_init=99.0,
    z_final=0.0,
    omega_m=0.3,
    seed=42,
):
    """Run a simple N-body simulation.

    Parameters
    ----------
    n_particles : int
        Number of particles per dimension (total particles = n_particles^3)
    box_size : float
        Box size in Mpc/h
    mesh_shape : int
        Number of grid cells per dimension for density field
    z_init : float
        Initial redshift
    z_final : float
        Final redshift
    omega_m : float
        Matter density parameter
    seed : int
        Random seed for initial conditions

    Returns
    -------
    dict
        Dictionary with:
        - 'positions_init': Initial particle positions
        - 'positions_final': Final particle positions
        - 'density_init': Initial density field
        - 'density_final': Final density field
        - 'box_size': Box size
        - 'mesh_shape': Mesh shape
    """
    print(f"Running N-body simulation:")
    print(f"  Particles: {n_particles}^3 = {n_particles**3}")
    print(f"  Box size: {box_size} Mpc/h")
    print(f"  Mesh: {mesh_shape}^3")
    print(f"  Redshift: {z_init} -> {z_final}")
    print(f"  Omega_m: {omega_m}")

    key = jax.random.PRNGKey(seed)

    # Create uniform grid of particles
    positions_init = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, box_size, n_particles, endpoint=False),
            jnp.linspace(0, box_size, n_particles, endpoint=False),
            jnp.linspace(0, box_size, n_particles, endpoint=False),
            indexing='ij'
        ),
        axis=-1
    ).reshape(-1, 3)

    # Add small random perturbations
    key, subkey = jax.random.split(key)
    perturbations = jax.random.normal(subkey, positions_init.shape) * (box_size / n_particles / 10)
    positions_init = positions_init + perturbations

    # Wrap positions to stay in box
    positions_init = jnp.mod(positions_init, box_size)

    # Paint initial density field
    # Convert positions to mesh coordinates
    mesh_positions_init = positions_init * (mesh_shape / box_size)
    # Split into integer mesh IDs and fractional displacements
    pmid_init = jnp.floor(mesh_positions_init).astype(jnp.int32)
    disp_init = mesh_positions_init - pmid_init

    density_init = scatter(
        pmid_init,
        disp_init,
        jnp.zeros([mesh_shape, mesh_shape, mesh_shape]),
        val=1.0,
        cell_size=1.0
    )

    # Simple test evolution: add large-scale coherent displacement to show structure
    # This is a placeholder - proper N-body evolution will be implemented later
    # Create a simple displacement field to show clustering
    displacement_amplitude = box_size / n_particles * 2.0  # Move particles ~2 grid cells

    # Add coherent displacement pattern to create structure
    key, subkey = jax.random.split(key)
    coherent_displacement = jax.random.normal(subkey, (n_particles, n_particles, n_particles, 3)) * displacement_amplitude
    coherent_displacement = coherent_displacement.reshape(-1, 3)

    positions_final = positions_init + coherent_displacement
    positions_final = jnp.mod(positions_final, box_size)

    mesh_positions_final = positions_final * (mesh_shape / box_size)
    pmid_final = jnp.floor(mesh_positions_final).astype(jnp.int32)
    disp_final = mesh_positions_final - pmid_final

    # Paint final density field
    density_final = scatter(
        pmid_final,
        disp_final,
        jnp.zeros([mesh_shape, mesh_shape, mesh_shape]),
        val=1.0,
        cell_size=1.0
    )

    return {
        'positions_init': positions_init,
        'positions_final': positions_final,
        'density_init': density_init,
        'density_final': density_final,
        'box_size': box_size,
        'mesh_shape': mesh_shape,
        'z_init': z_init,
        'z_final': z_final,
    }
