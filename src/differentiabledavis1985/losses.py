"""
Loss functions for evaluating jaxpm-based reconstructions.

This module provides chi-square based loss functions for comparing
simulated and target density fields or particle distributions.
"""

import jax.numpy as jnp


def chi2(input_arr, target_arr, sigma=None):
    """
    Compute chi-squared loss between input and target arrays.

    Parameters
    ----------
    input_arr : jnp.ndarray
        Input/predicted array (e.g., reconstructed density field)
    target_arr : jnp.ndarray
        Target/reference array (e.g., true density field)
    sigma : jnp.ndarray, optional
        Uncertainty/noise level for each element. If provided, will weight
        the residuals by 1/sigma. Can be scalar or array matching input shape.

    Returns
    -------
    float
        Chi-squared value: sum of squared weighted residuals

    Examples
    --------
    >>> input_field = jnp.array([1.0, 2.0, 3.0])
    >>> target_field = jnp.array([1.1, 2.2, 2.9])
    >>> loss = chi2(input_field, target_field)

    >>> # With uncertainties
    >>> sigma = jnp.array([0.1, 0.1, 0.1])
    >>> loss_weighted = chi2(input_field, target_field, sigma=sigma)
    """
    delta = input_arr - target_arr
    if sigma is not None:
        delta /= sigma
    return jnp.sum(delta ** 2)


def chi2_log(input_arr, target_arr, epsilon=1e-10, sigma=None):
    """
    Compute chi-squared loss in log-space between input and target arrays.

    This is useful for comparing quantities that span many orders of magnitude,
    such as density fields where both underdense and overdense regions are
    important.

    Parameters
    ----------
    input_arr : jnp.ndarray
        Input/predicted array (e.g., reconstructed density field)
    target_arr : jnp.ndarray
        Target/reference array (e.g., true density field)
    epsilon : float, optional
        Small constant added before taking logarithm to avoid log(0).
        Default is 1e-10.
    sigma : jnp.ndarray, optional
        Uncertainty/noise level for each element in linear space. If provided,
        will be converted to log-space and used to weight the residuals.
        Can be scalar or array matching input shape.

    Returns
    -------
    float
        Chi-squared value in log-space: sum of squared weighted log residuals

    Examples
    --------
    >>> input_field = jnp.array([1.0, 10.0, 100.0])
    >>> target_field = jnp.array([1.1, 11.0, 110.0])
    >>> loss = chi2_log(input_field, target_field)

    >>> # With uncertainties in linear space
    >>> sigma = jnp.array([0.1, 1.0, 10.0])
    >>> loss_weighted = chi2_log(input_field, target_field, sigma=sigma)
    """
    arr1 = jnp.log10(input_arr + epsilon)
    arr2 = jnp.log10(target_arr + epsilon)
    delta = arr1 - arr2
    if sigma is not None:
        # Convert linear-space uncertainty to log-space
        sigma_log = sigma / (target_arr + epsilon)
        delta /= sigma_log
    return jnp.sum(delta ** 2)


def reduced_chi2(input_arr, target_arr, sigma=None, n_dof=None):
    """
    Compute reduced chi-squared (chi^2 / n_dof).

    Parameters
    ----------
    input_arr : jnp.ndarray
        Input/predicted array
    target_arr : jnp.ndarray
        Target/reference array
    sigma : jnp.ndarray, optional
        Uncertainty/noise level
    n_dof : int, optional
        Number of degrees of freedom. If None, uses the total number of
        elements in the arrays.

    Returns
    -------
    float
        Reduced chi-squared value
    """
    chi2_val = chi2(input_arr, target_arr, sigma=sigma)
    if n_dof is None:
        n_dof = input_arr.size
    return chi2_val / n_dof


def reduced_chi2_log(input_arr, target_arr, epsilon=1e-10, sigma=None, n_dof=None):
    """
    Compute reduced chi-squared in log-space (chi^2_log / n_dof).

    Parameters
    ----------
    input_arr : jnp.ndarray
        Input/predicted array
    target_arr : jnp.ndarray
        Target/reference array
    epsilon : float, optional
        Small constant added before taking logarithm
    sigma : jnp.ndarray, optional
        Uncertainty/noise level in linear space
    n_dof : int, optional
        Number of degrees of freedom. If None, uses the total number of
        elements in the arrays.

    Returns
    -------
    float
        Reduced chi-squared value in log-space
    """
    chi2_val = chi2_log(input_arr, target_arr, epsilon=epsilon, sigma=sigma)
    if n_dof is None:
        n_dof = input_arr.size
    return chi2_val / n_dof
