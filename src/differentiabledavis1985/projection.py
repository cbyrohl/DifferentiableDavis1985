"""
Differentiable projection functions for mapping 3D density/particles to 2D masks.

This module provides JAX-differentiable operations for:
- Projecting 3D density fields to 2D
- Converting 2D projections to soft occupancy masks
- Comparing simulated projections to observed 2D masks

The main workflow is:
1. Run simulation to get 3D particle positions
2. Paint positions to 3D density using CIC (done by forward_model)
3. Project 3D density to 2D by summing along one axis
4. Convert to soft mask using soft_occupancy or soft_threshold
5. Compare to target mask using loss functions
"""

import jax
import jax.numpy as jnp
from typing import Optional


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
    scale: float = 1.0,
    normalize: bool = True
) -> jnp.ndarray:
    """
    Convert a 3D density field to a 2D soft mask.

    This is the main function for differentiable projection. It:
    1. Projects the 3D density to 2D by summing along an axis
    2. Optionally normalizes by depth to prevent saturation
    3. Applies a soft thresholding function to create a mask-like output

    Args:
        density_3d: 3D density field of shape (nx, ny, nz)
        axis: Axis to project along (0=x, 1=y, 2=z)
        threshold: Threshold for sigmoid method (auto-computed if None)
        sharpness: Sharpness for sigmoid method
        method: "sigmoid" for threshold-based, "occupancy" for exponential
        scale: Scale factor for occupancy method
        normalize: If True, normalize projection by depth to prevent saturation

    Returns:
        2D soft mask in [0, 1] range
    """
    # Project to 2D
    projection = project_density_2d(density_3d, axis=axis)

    # Normalize by depth to keep values in reasonable range
    if normalize:
        depth = density_3d.shape[axis]
        projection = projection / depth

    if method == "sigmoid":
        if threshold is None:
            # Auto threshold at mean value
            threshold = jnp.mean(projection)
        return soft_threshold(projection, threshold=threshold, sharpness=sharpness)
    elif method == "occupancy":
        return soft_occupancy(projection, scale=scale)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sigmoid' or 'occupancy'.")


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
