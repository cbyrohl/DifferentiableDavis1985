"""
Forward model for reproducing Davis et al. 1985 N-body simulations.

This module implements differentiable gravitational simulations using jaxpm
to reproduce the particle distributions from the seminal CDM paper.
"""

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn
from typing import Optional, Tuple

jax.config.update('jax_enable_x64', True)


class BaseSimulation:
    """Base class for cosmological N-body simulations."""

    def __init__(self, mesh_shape: Tuple[int, int, int], box_size: float, snapshots: jnp.ndarray):
        """
        Initialize base simulation parameters.

        Args:
            mesh_shape: Grid dimensions (nx, ny, nz)
            box_size: Comoving box size in Mpc/h
            snapshots: Array of scale factors for output snapshots
        """
        self.mesh_shape = mesh_shape
        # Convert box_size to array for jaxpm
        self.box_size = jnp.array([box_size, box_size, box_size])
        self.snapshots = snapshots

    def run_simulation(self, initial_conditions: jnp.ndarray):
        """Run the N-body simulation. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this!")


class Davis1985Simulation(BaseSimulation):
    """
    Gravitational N-body simulation for reproducing Davis et al. 1985.

    Implements cold dark matter simulations with:
    - Adiabatic initial conditions
    - Constant curvature primordial spectrum
    - Particle-mesh gravity solver via jaxpm
    """

    def __init__(
        self,
        mesh_shape: Tuple[int, int, int],
        box_size: float,
        snapshots: jnp.ndarray,
        omega_c: float = 0.25,
        omega_b: float = 0.045,
        sigma8: float = 0.8,
        h: float = 0.7,
        n_s: float = 1.0,
    ):
        """
        Initialize Davis 1985 simulation.

        Args:
            mesh_shape: Grid dimensions (nx, ny, nz). Davis85 used 32^3 particles
            box_size: Comoving box size in Mpc/h
            snapshots: Array of scale factors for outputs
            omega_c: Cold dark matter density parameter
            omega_b: Baryon density parameter
            sigma8: Amplitude of matter fluctuations
            h: Hubble parameter H0 / (100 km/s/Mpc)
            n_s: Scalar spectral index
        """
        super().__init__(mesh_shape, box_size, snapshots)

        self.omega_c = omega_c
        self.omega_b = omega_b
        self.sigma8 = sigma8
        self.h = h
        self.n_s = n_s

        # Initialize cosmology (Davis85 primarily used Omega=1 and Omega=0.2 models)
        self.cosmo = jc.Planck15(
            Omega_c=omega_c,
            Omega_b=omega_b,
            sigma8=sigma8,
            h=h,
            n_s=n_s
        )

        self.pk = None
        self.pk_fn = None

    def _initialize_power_spectrum(self):
        """
        Initialize the linear matter power spectrum.

        Davis et al. 1985 used a constant curvature spectrum tilting from
        n=1 (large scales) to n=-3 (small scales).
        """
        k = jnp.logspace(-4, 1, 128)
        # Create temporary cosmo to avoid jaxpm caching issues
        cosmo = jc.Planck15(
            Omega_c=self.omega_c,
            Omega_b=self.omega_b,
            sigma8=self.sigma8,
            h=self.h,
            n_s=self.n_s
        )
        self.pk = jc.power.linear_matter_power(cosmo, k)
        self.pk_fn = lambda x: jnp.interp(
            x.reshape([-1]), k, self.pk
        ).reshape(x.shape)

    def generate_initial_conditions(self, seed: int = 4242) -> jnp.ndarray:
        """
        Generate random Gaussian initial conditions.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial density field on mesh
        """
        if self.pk_fn is None:
            self._initialize_power_spectrum()

        return linear_field(
            self.mesh_shape,
            self.box_size,
            self.pk_fn,
            seed=jax.random.PRNGKey(seed)
        )

    def generate_zero_initial_conditions(self) -> jnp.ndarray:
        """
        Generate zero initial conditions (for testing).

        Returns:
            Zero-filled density field
        """
        return jnp.zeros(self.mesh_shape)

    def run_simulation(
        self,
        initial_conditions: jnp.ndarray,
        rtol: float = 1e-8,
        atol: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the gravitational N-body simulation.

        Uses Lagrangian Perturbation Theory (LPT) for initial displacement
        and particle-mesh integration for time evolution.

        Args:
            initial_conditions: Initial density field
            rtol: Relative tolerance for ODE integrator
            atol: Absolute tolerance for ODE integrator

        Returns:
            Tuple of (particle_positions, initial_positions)
            - particle_positions: Array of particle positions at each snapshot
            - initial_positions: Initial displaced particle positions
        """
        # Initialize particles on regular grid (like Davis et al. 1985)
        particles = jnp.stack(
            jnp.meshgrid(*[jnp.arange(s) for s in self.mesh_shape]),
            axis=-1
        ).reshape([-1, 3])

        # Apply Lagrangian perturbation theory for initial displacement
        dx, p, f = lpt(
            self.cosmo,
            initial_conditions,
            particles,
            a=self.snapshots[0]
        )

        # Evolve particles using ODE integrator with PM forces
        result = odeint(
            make_ode_fn(self.mesh_shape),
            [particles + dx, p],
            self.snapshots,
            self.cosmo,
            rtol=rtol,
            atol=atol
        )

        # Return evolved positions and initial displaced positions
        return result[0], particles + dx

    def paint_density(
        self,
        positions: jnp.ndarray,
        mesh_shape: Optional[Tuple[int, int, int]] = None
    ) -> jnp.ndarray:
        """
        Paint particle positions to density field using CIC.

        Args:
            positions: Particle positions
            mesh_shape: Mesh shape (defaults to self.mesh_shape)

        Returns:
            Density field on mesh
        """
        if mesh_shape is None:
            mesh_shape = self.mesh_shape

        return cic_paint(jnp.zeros(mesh_shape), positions)
