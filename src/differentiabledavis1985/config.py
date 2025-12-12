"""Configuration management for simulations."""

from pathlib import Path
from typing import Optional
import yaml


class SimulationConfig:
    """Configuration for N-body simulation parameters."""

    def __init__(
        self,
        n_particles: int = 32,
        omega_m: float = 0.3,
        a_init: float = 0.01,
        a_final: float = 1.0,
        box_size: float = 100.0,
        mesh_shape: Optional[int] = None,
        seed: int = 42,
    ):
        self.n_particles = n_particles
        self.omega_m = omega_m
        self.a_init = a_init
        self.a_final = a_final
        self.box_size = box_size
        self.mesh_shape = mesh_shape if mesh_shape is not None else n_particles // 2
        self.seed = seed

    @property
    def z_init(self) -> float:
        """Convert initial scale factor to redshift."""
        return 1.0 / self.a_init - 1.0

    @property
    def z_final(self) -> float:
        """Convert final scale factor to redshift."""
        return 1.0 / self.a_final - 1.0

    @classmethod
    def from_yaml(cls, config_path: str) -> "SimulationConfig":
        """Load configuration from YAML file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file

        Returns
        -------
        SimulationConfig
            Configuration object
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "n_particles": self.n_particles,
            "omega_m": self.omega_m,
            "a_init": self.a_init,
            "a_final": self.a_final,
            "box_size": self.box_size,
            "mesh_shape": self.mesh_shape,
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        return (
            f"SimulationConfig(n_particles={self.n_particles}, "
            f"omega_m={self.omega_m}, "
            f"a_init={self.a_init} (z={self.z_init:.1f}), "
            f"a_final={self.a_final} (z={self.z_final:.1f}), "
            f"box_size={self.box_size}, "
            f"mesh_shape={self.mesh_shape}, "
            f"seed={self.seed})"
        )
