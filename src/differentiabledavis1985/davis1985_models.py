"""
Specialized simulation configurations matching Davis et al. 1985 models.

This module provides pre-configured simulations matching the specific
models from the paper (EdS1-5, O1-4, etc.)
"""

import jax.numpy as jnp
from differentiabledavis1985.forward_model import Davis1985Simulation


class Fig1BottomLeftSimulation(Davis1985Simulation):
    """
    Reproduces Figure 1 (bottom left) from Davis et al. 1985.

    This corresponds to model EdS1 at expansion factor a = 2.4.
    An Einstein-de Sitter universe (Ω = 1) with cold dark matter.

    Parameters from Table 1 of Davis et al. 1985:
    - Box size L = 32.5 h^-2 Mpc
    - 32,768 particles (32^3 grid)
    - Ω = 1 (Einstein-de Sitter)
    - Final output at a = 2.4
    """

    def __init__(
        self,
        initial_a: float = 0.1,
        final_a: float = 2.4,
        n_snapshots: int = 5,
        seed: int = 42,
    ):
        """
        Initialize Fig1 bottom-left simulation.

        Args:
            initial_a: Initial expansion factor (default 0.1, early enough for linear regime)
            final_a: Final expansion factor (2.4 for Fig1 bottom-left)
            n_snapshots: Number of output snapshots
            seed: Random seed for initial conditions
        """
        # Generate snapshot times from initial to final
        snapshots = jnp.linspace(initial_a, final_a, n_snapshots)

        # Einstein-de Sitter universe parameters
        # Omega_m = Omega_c + Omega_b = 1.0
        # For simplicity, assume most mass is CDM
        omega_c = 0.95  # Cold dark matter
        omega_b = 0.05  # Baryons

        # Box size from Table 1: L = 32.5 h^-2 Mpc
        # We'll use h=1 as a reference (actual h cancels in many calculations)
        box_size = 32.5  # in units of h^-1 Mpc when h=1

        # 32^3 particles matching the original paper
        mesh_shape = (32, 32, 32)

        # Initialize with standard cosmological parameters
        # Davis et al. used sigma8 normalization - we'll use a typical value
        super().__init__(
            mesh_shape=mesh_shape,
            box_size=box_size,
            snapshots=snapshots,
            omega_c=omega_c,
            omega_b=omega_b,
            sigma8=0.8,  # Typical normalization
            h=1.0,  # Hubble parameter in units of 100 km/s/Mpc
            n_s=1.0,  # Harrison-Zel'dovich spectrum (constant curvature)
        )

        self.seed = seed
        self.initial_a = initial_a
        self.final_a = final_a

    def run(self):
        """
        Run the complete simulation workflow.

        Returns:
            Tuple of (particle_positions, initial_positions)
            where particle_positions has shape (n_snapshots, n_particles, 3)
        """
        # Generate initial conditions with the specified seed
        initial_conditions = self.generate_initial_conditions(seed=self.seed)

        # Run the N-body simulation
        positions, initial_positions = self.run_simulation(initial_conditions)

        return positions, initial_positions

    def get_final_snapshot(self):
        """
        Get only the final snapshot at a = 2.4.

        Returns:
            Final particle positions at a = 2.4
        """
        positions, _ = self.run()
        return positions[-1]  # Return last snapshot


class EdS1Simulation(Davis1985Simulation):
    """
    Full EdS1 model evolution from Davis et al. 1985.

    Einstein-de Sitter model with Ω = 1, evolved to multiple output times
    as shown in Figure 1 (a = 1.8, 2.4, 4.5) and used in the analysis.
    """

    def __init__(
        self,
        output_times: list = None,
        seed: int = 42,
    ):
        """
        Initialize EdS1 simulation.

        Args:
            output_times: List of expansion factors for outputs
                         (default matches Figure 1: [1.0, 1.8, 2.4, 4.5])
            seed: Random seed for initial conditions
        """
        if output_times is None:
            # Default outputs matching Figure 1 and analysis in paper
            output_times = [0.1, 1.0, 1.4, 1.8, 2.4, 3.0, 3.5, 4.5]

        snapshots = jnp.array(output_times)

        # Einstein-de Sitter parameters
        omega_c = 0.95
        omega_b = 0.05
        box_size = 32.5
        mesh_shape = (32, 32, 32)

        super().__init__(
            mesh_shape=mesh_shape,
            box_size=box_size,
            snapshots=snapshots,
            omega_c=omega_c,
            omega_b=omega_b,
            sigma8=0.8,
            h=1.0,
            n_s=1.0,
        )

        self.seed = seed
        self.output_times = output_times

    def run(self):
        """Run the EdS1 simulation."""
        initial_conditions = self.generate_initial_conditions(seed=self.seed)
        return self.run_simulation(initial_conditions)
