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
        suffix: Optional[str] = None,
        n_steps_min: int = 2,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        reconstruction_seed: int = 1234,
        reconstruction_init_scale: float = 0.1,
        n_iterations: int = 100,
    ):
        self.n_particles = n_particles
        self.omega_m = omega_m
        self.a_init = a_init
        self.a_final = a_final
        self.box_size = box_size
        self.mesh_shape = mesh_shape if mesh_shape is not None else n_particles // 2
        self.seed = seed
        self.suffix = suffix
        self.n_steps_min = n_steps_min
        self.rtol = rtol
        self.atol = atol
        self.reconstruction_seed = reconstruction_seed
        self.reconstruction_init_scale = reconstruction_init_scale
        self.n_iterations = n_iterations

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

        # Derive suffix from config filename if not explicitly set
        if "suffix" not in config_dict or config_dict["suffix"] is None:
            config_dict["suffix"] = path.stem

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
            "suffix": self.suffix,
            "n_steps_min": self.n_steps_min,
            "rtol": self.rtol,
            "atol": self.atol,
            "reconstruction_seed": self.reconstruction_seed,
            "reconstruction_init_scale": self.reconstruction_init_scale,
            "n_iterations": self.n_iterations,
        }

    def __repr__(self) -> str:
        return (
            f"SimulationConfig(n_particles={self.n_particles}, "
            f"omega_m={self.omega_m}, "
            f"a_init={self.a_init} (z={self.z_init:.1f}), "
            f"a_final={self.a_final} (z={self.z_final:.1f}), "
            f"box_size={self.box_size}, "
            f"mesh_shape={self.mesh_shape}, "
            f"seed={self.seed}, "
            f"n_steps_min={self.n_steps_min}, "
            f"rtol={self.rtol}, atol={self.atol}, "
            f"reconstruction_seed={self.reconstruction_seed}, "
            f"reconstruction_init_scale={self.reconstruction_init_scale}, "
            f"n_iterations={self.n_iterations})"
        )
