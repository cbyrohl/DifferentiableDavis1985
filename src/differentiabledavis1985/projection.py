"""
Differentiable projection functions for mapping 3D density/particles to 2D masks.

This module provides JAX-differentiable operations for:
- Projecting 3D density fields to 2D
- Converting particle positions to 2D soft occupancy masks
- Comparing simulated projections to observed 2D masks
"""

import jax
import jax.numpy as jnp
from jaxpm.painting import cic_paint
from typing import Tuple, Optional


def project_density_2d(
    density_3d: jnp.ndarray,
    axis: int = 2
) -> jnp.ndarray:
    """
    Project a 3D density field to 2D by summing along one axis.

    This creates a column density map (surface density) analogous to
    what would be observed in a scatter plot projection.

    Args:
        density_3d: 3D density field of shape (nx, ny, nz)
        axis: Axis to project along (0=x, 1=y, 2=z). Default z.

    Returns:
        2D projected density of shape (n1, n2) where n1, n2 are the
        dimensions perpendicular to the projection axis.
    """
    return jnp.sum(density_3d, axis=axis)


def soft_threshold(
    x: jnp.ndarray,
    threshold: float = 0.5,
    sharpness: float = 10.0
) -> jnp.ndarray:
    """
    Apply a differentiable soft threshold (sigmoid) to an array.

    Maps values to [0, 1] range using:
        sigmoid((x - threshold) * sharpness)

    Args:
        x: Input array
        threshold: Center of the sigmoid transition
        sharpness: Steepness of transition (higher = sharper)

    Returns:
        Soft-thresholded array in [0, 1] range
    """
    return jax.nn.sigmoid((x - threshold) * sharpness)


def soft_occupancy(
    x: jnp.ndarray,
    scale: float = 1.0
) -> jnp.ndarray:
    """
    Convert density/counts to soft occupancy using exponential saturation.

    Maps values to [0, 1] using:
        1 - exp(-x * scale)

    This gives:
    - 0 when x = 0
    - Approaches 1 as x increases
    - Smooth gradients everywhere

    Args:
        x: Input array (e.g., projected density or particle counts)
        scale: Scale factor controlling saturation rate

    Returns:
        Soft occupancy in [0, 1] range
    """
    return 1.0 - jnp.exp(-x * scale)


def density_to_soft_mask(
    density_3d: jnp.ndarray,
    axis: int = 2,
    threshold: Optional[float] = None,
    sharpness: float = 10.0,
    method: str = "occupancy",
    scale: float = 1.0
) -> jnp.ndarray:
    """
    Convert a 3D density field to a 2D soft mask.

    This is the main function for differentiable projection. It:
    1. Projects the 3D density to 2D by summing along an axis
    2. Applies a soft thresholding function to create a mask-like output

    Args:
        density_3d: 3D density field of shape (nx, ny, nz)
        axis: Axis to project along (0=x, 1=y, 2=z)
        threshold: Threshold for sigmoid method (auto-computed if None)
        sharpness: Sharpness for sigmoid method
        method: "sigmoid" for threshold-based, "occupancy" for exponential
        scale: Scale factor for occupancy method

    Returns:
        2D soft mask in [0, 1] range
    """
    # Project to 2D
    projection = project_density_2d(density_3d, axis=axis)

    if method == "sigmoid":
        if threshold is None:
            # Auto threshold at mean value
            threshold = jnp.mean(projection)
        return soft_threshold(projection, threshold=threshold, sharpness=sharpness)
    elif method == "occupancy":
        return soft_occupancy(projection, scale=scale)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sigmoid' or 'occupancy'.")


def positions_to_2d_density(
    positions: jnp.ndarray,
    mesh_shape_2d: Tuple[int, int],
    mesh_shape_3d: Tuple[int, int, int],
    axis: int = 2
) -> jnp.ndarray:
    """
    Paint particle positions directly to a 2D density field.

    Projects 3D positions to 2D and uses CIC painting to create
    a smooth density field.

    Args:
        positions: Particle positions of shape (N, 3) in mesh units
        mesh_shape_2d: Output 2D mesh shape (nx, ny)
        mesh_shape_3d: Original 3D mesh shape (for coordinate scaling)
        axis: Axis to project along (0=x, 1=y, 2=z)

    Returns:
        2D density field of shape mesh_shape_2d
    """
    # Extract 2D coordinates by dropping the projection axis
    if axis == 0:
        pos_2d = positions[:, [1, 2]]  # (y, z)
    elif axis == 1:
        pos_2d = positions[:, [0, 2]]  # (x, z)
    else:  # axis == 2
        pos_2d = positions[:, [0, 1]]  # (x, y)

    # Scale coordinates if mesh shapes differ
    scale_factors = jnp.array([
        mesh_shape_2d[0] / mesh_shape_3d[0 if axis != 0 else 1],
        mesh_shape_2d[1] / mesh_shape_3d[2 if axis != 2 else 1]
    ])

    if axis == 0:
        scale_factors = jnp.array([
            mesh_shape_2d[0] / mesh_shape_3d[1],
            mesh_shape_2d[1] / mesh_shape_3d[2]
        ])
    elif axis == 1:
        scale_factors = jnp.array([
            mesh_shape_2d[0] / mesh_shape_3d[0],
            mesh_shape_2d[1] / mesh_shape_3d[2]
        ])
    else:
        scale_factors = jnp.array([
            mesh_shape_2d[0] / mesh_shape_3d[0],
            mesh_shape_2d[1] / mesh_shape_3d[1]
        ])

    pos_2d_scaled = pos_2d * scale_factors

    # Paint to 2D grid using CIC
    # Note: cic_paint expects 3D, so we use a workaround with a thin 3D grid
    mesh_3d_thin = (mesh_shape_2d[0], mesh_shape_2d[1], 1)
    pos_3d_thin = jnp.concatenate([pos_2d_scaled, jnp.zeros((positions.shape[0], 1))], axis=1)

    density_3d = cic_paint(jnp.zeros(mesh_3d_thin), pos_3d_thin)
    return density_3d[:, :, 0]


def positions_to_soft_mask(
    positions: jnp.ndarray,
    mask_shape: Tuple[int, int],
    mesh_shape_3d: Tuple[int, int, int],
    axis: int = 2,
    method: str = "occupancy",
    scale: float = 1.0,
    threshold: Optional[float] = None,
    sharpness: float = 10.0
) -> jnp.ndarray:
    """
    Convert 3D particle positions to a 2D soft mask.

    This is the primary function for comparing simulations to observed masks.
    It projects particle positions to 2D and creates a differentiable
    approximation of particle occupancy.

    Args:
        positions: Particle positions of shape (N, 3) in mesh units
        mask_shape: Output mask shape (nx, ny)
        mesh_shape_3d: Original 3D mesh shape for coordinate scaling
        axis: Axis to project along (0=x, 1=y, 2=z)
        method: "occupancy" (exponential) or "sigmoid" (threshold-based)
        scale: Scale for occupancy method
        threshold: Threshold for sigmoid method
        sharpness: Sharpness for sigmoid method

    Returns:
        2D soft mask in [0, 1] range of shape mask_shape
    """
    # Paint positions to 2D density
    density_2d = positions_to_2d_density(
        positions, mask_shape, mesh_shape_3d, axis=axis
    )

    # Convert to soft mask
    if method == "occupancy":
        return soft_occupancy(density_2d, scale=scale)
    elif method == "sigmoid":
        if threshold is None:
            threshold = jnp.mean(density_2d)
        return soft_threshold(density_2d, threshold=threshold, sharpness=sharpness)
    else:
        raise ValueError(f"Unknown method: {method}")


def mask_loss_bce(
    predicted_mask: jnp.ndarray,
    target_mask: jnp.ndarray,
    epsilon: float = 1e-7
) -> jnp.ndarray:
    """
    Binary cross-entropy loss for mask comparison.

    Args:
        predicted_mask: Predicted soft mask in [0, 1]
        target_mask: Target binary mask (0 or 1)
        epsilon: Small constant for numerical stability

    Returns:
        BCE loss (scalar)
    """
    # Clamp predictions to avoid log(0)
    pred_clamped = jnp.clip(predicted_mask, epsilon, 1.0 - epsilon)

    # Binary cross-entropy
    bce = -target_mask * jnp.log(pred_clamped) - (1 - target_mask) * jnp.log(1 - pred_clamped)

    return jnp.mean(bce)


def mask_loss_mse(
    predicted_mask: jnp.ndarray,
    target_mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Mean squared error loss for mask comparison.

    Args:
        predicted_mask: Predicted soft mask in [0, 1]
        target_mask: Target binary mask (0 or 1)

    Returns:
        MSE loss (scalar)
    """
    return jnp.mean((predicted_mask - target_mask) ** 2)


def mask_loss_dice(
    predicted_mask: jnp.ndarray,
    target_mask: jnp.ndarray,
    smooth: float = 1.0
) -> jnp.ndarray:
    """
    Dice loss for mask comparison (good for imbalanced masks).

    The Dice coefficient measures overlap between predicted and target.
    Loss = 1 - Dice, so minimizing gives maximum overlap.

    Args:
        predicted_mask: Predicted soft mask in [0, 1]
        target_mask: Target binary mask (0 or 1)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice loss (scalar, lower is better)
    """
    intersection = jnp.sum(predicted_mask * target_mask)
    union = jnp.sum(predicted_mask) + jnp.sum(target_mask)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def mask_loss_focal(
    predicted_mask: jnp.ndarray,
    target_mask: jnp.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    epsilon: float = 1e-7
) -> jnp.ndarray:
    """
    Focal loss for mask comparison (handles class imbalance well).

    Focal loss down-weights easy examples and focuses on hard ones.

    Args:
        predicted_mask: Predicted soft mask in [0, 1]
        target_mask: Target binary mask (0 or 1)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing factor for positive class
        epsilon: Small constant for numerical stability

    Returns:
        Focal loss (scalar)
    """
    pred_clamped = jnp.clip(predicted_mask, epsilon, 1.0 - epsilon)

    # Compute focal weights
    p_t = target_mask * pred_clamped + (1 - target_mask) * (1 - pred_clamped)
    focal_weight = (1 - p_t) ** gamma

    # Compute BCE
    bce = -target_mask * jnp.log(pred_clamped) - (1 - target_mask) * jnp.log(1 - pred_clamped)

    # Apply alpha balancing
    alpha_t = target_mask * alpha + (1 - target_mask) * (1 - alpha)

    return jnp.mean(alpha_t * focal_weight * bce)
