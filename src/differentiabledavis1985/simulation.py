"""Simple N-body simulation runner using jaxpm."""

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.painting import cic_paint
from jaxpm.growth import growth_factor
from loguru import logger


def run_nbody_simulation(
    n_particles=32,
    box_size=100.0,
    mesh_shape=64,
    z_init=99.0,
    z_final=0.0,
    omega_m=0.3,
    seed=42,
    n_steps_min=2,
    rtol=1e-5,
    atol=1e-5,
):
    """Run an N-body simulation using jaxpm.

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
    n_steps_min : int
        Minimum number of output snapshots for ODE integrator (default: 2)
    rtol : float
        Relative tolerance for ODE integrator (default: 1e-5)
    atol : float
        Absolute tolerance for ODE integrator (default: 1e-5)

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
    logger.info("Running N-body simulation")
    logger.info(f"  Particles: {n_particles}^3 = {n_particles**3}")
    logger.info(f"  Box size: {box_size} Mpc/h")
    logger.info(f"  Mesh: {mesh_shape}^3")
    logger.info(f"  Redshift: {z_init} -> {z_final}")
    logger.info(f"  Omega_m: {omega_m}")
    logger.info(f"  n_steps_min: {n_steps_min}")
    logger.info(f"  rtol: {rtol}, atol: {atol}")

    # Convert redshift to scale factor
    a_init = 1.0 / (1.0 + z_init)
    a_final = 1.0 / (1.0 + z_final)
    snapshots = jnp.linspace(a_init, a_final, n_steps_min)
    logger.debug(f"Scale factors: a_init={a_init:.6f}, a_final={a_final:.6f}, n_snapshots={len(snapshots)}")

    # Set up mesh and box arrays for jaxpm
    mesh_shape_3d = (mesh_shape, mesh_shape, mesh_shape)
    box_size_3d = jnp.array([box_size, box_size, box_size])

    # Create cosmology - split omega_m into CDM and baryons
    omega_b = 0.045
    omega_c = omega_m - omega_b
    logger.debug(f"Cosmology: omega_c={omega_c:.4f}, omega_b={omega_b:.4f}")

    cosmo = jc.Planck15(
        Omega_c=omega_c,
        Omega_b=omega_b,
        sigma8=0.8,
        h=0.7,
        n_s=1.0
    )

    # Precompute growth factor to populate jaxpm workspace cache
    # (must be done before power spectrum to avoid cache corruption)
    _ = growth_factor(cosmo, jnp.atleast_1d(1.0))

    # Generate linear matter power spectrum (use fresh cosmo to avoid cache issues)
    logger.debug("Generating power spectrum")
    k = jnp.logspace(-4, 1, 128)
    cosmo_pk = jc.Planck15(
        Omega_c=omega_c,
        Omega_b=omega_b,
        sigma8=0.8,
        h=0.7,
        n_s=1.0
    )
    pk = jc.power.linear_matter_power(cosmo_pk, k)
    pk_fn = lambda x: jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Generate Gaussian random initial conditions
    logger.debug(f"Generating initial conditions with seed={seed}")
    key = jax.random.PRNGKey(seed)
    initial_conditions = linear_field(mesh_shape_3d, box_size_3d, pk_fn, seed=key)
    logger.debug(f"Initial conditions shape: {initial_conditions.shape}")
    logger.debug(f"Initial conditions: min={float(initial_conditions.min()):.4f}, "
                 f"mean={float(initial_conditions.mean()):.4f}, max={float(initial_conditions.max()):.4f}")

    # Create particle grid (in mesh units, like jaxpm expects)
    logger.debug("Creating particle grid")
    particles = jnp.stack(
        jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape_3d]),
        axis=-1
    ).reshape([-1, 3])
    logger.debug(f"Particle grid shape: {particles.shape}")

    # Apply Lagrangian perturbation theory for initial displacement
    logger.debug("Applying LPT for initial displacement")
    dx, p, f = lpt(cosmo, initial_conditions, particles, a=a_init)
    positions_init = particles + dx
    logger.debug(f"Initial displaced positions: min={float(positions_init.min()):.4f}, "
                 f"max={float(positions_init.max()):.4f}")

    # Paint initial density field
    logger.debug("Painting initial density field")
    density_init = cic_paint(jnp.zeros(mesh_shape_3d), positions_init)
    logger.debug(f"Initial density: min={float(density_init.min()):.4f}, "
                 f"mean={float(density_init.mean()):.4f}, max={float(density_init.max()):.4f}")

    # Evolve particles using ODE integrator with PM forces
    logger.info("Evolving particles with PM gravity solver...")
    result = odeint(
        make_ode_fn(mesh_shape_3d),
        [positions_init, p],
        snapshots,
        cosmo,
        rtol=rtol,
        atol=atol
    )

    # Extract all positions (result[0] is positions at each snapshot)
    all_positions = result[0]  # All snapshots
    positions_final = all_positions[-1]  # Last snapshot
    logger.debug(f"Final positions: min={float(positions_final.min()):.4f}, "
                 f"max={float(positions_final.max()):.4f}")

    # Paint density fields for all snapshots
    logger.debug("Painting density fields for all snapshots")
    densities = []
    for i, pos in enumerate(all_positions):
        density = cic_paint(jnp.zeros(mesh_shape_3d), pos)
        densities.append(density)
        if i == len(all_positions) - 1:  # Log final density
            logger.debug(f"Final density: min={float(density.min()):.4f}, "
                        f"mean={float(density.mean()):.4f}, max={float(density.max()):.4f}")

    density_final = densities[-1]

    # Convert positions from mesh units to physical units for output
    cell_size = box_size / mesh_shape
    positions_init_physical = positions_init * cell_size
    positions_final_physical = positions_final * cell_size

    logger.info("Simulation complete")
    return {
        'positions_init': positions_init_physical,
        'positions_final': positions_final_physical,
        'density_init': density_init,
        'density_final': density_final,
        'densities': densities,  # All density snapshots
        'scale_factors': snapshots,  # All scale factors
        'box_size': box_size,
        'mesh_shape': mesh_shape,
        'z_init': z_init,
        'z_final': z_final,
    }
