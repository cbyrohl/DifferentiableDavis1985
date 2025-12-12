"""Plotting utilities for density field visualization."""

import numpy as np
import matplotlib.pyplot as plt


def plot_density_slice(slc, extent, imshow_kwargs=None):
    """Plot a 2D slice of a density field.

    Parameters
    ----------
    slc : array_like
        2D density slice to plot
    extent : list
        [xmin, xmax, ymin, ymax] extent for imshow
    imshow_kwargs : dict, optional
        Additional kwargs to pass to imshow

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    print("min/mean/max:", slc.min(), slc.mean(), slc.max())
    pmin = np.percentile(slc, 1)
    pmin = max(0.0001, pmin)
    pmax = np.percentile(slc, 99.9)

    if imshow_kwargs is None:
        imshow_kwargs = {}

    ikw = dict(
        cmap="viridis",
        interpolation="nearest",
        vmin=pmin,
        vmax=pmax,
        extent=extent,
        origin="lower",
        norm="log"
    )
    ikw.update(**imshow_kwargs)

    ax.imshow(slc.T, **ikw)
    ax.set_xlabel("cMpc/h")
    ax.set_ylabel("cMpc/h")
    cbar = plt.colorbar(ax.images[0], ax=ax)
    cbar.set_label(r"density [(cMpc/h)$^{-3}$]")

    return fig, ax


def plot_density_slice_from_cube(cube, boxsize, slc_idx=None, losidx=0, **kwargs):
    """Plot a 2D slice from a 3D density cube.

    Parameters
    ----------
    cube : array_like
        3D density cube
    boxsize : array_like or float
        Box size in each dimension. If float, assumes cubic box.
    slc_idx : int, optional
        Index of slice to plot. If None, plots middle slice.
    losidx : int
        Line-of-sight axis (0, 1, or 2)
    **kwargs
        Additional kwargs passed to plot_density_slice

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if isinstance(boxsize, (int, float)):
        boxsize = [boxsize] * 3

    if slc_idx is None:
        slc_idx = cube.shape[losidx] // 2

    if losidx == 0:
        slc = cube[slc_idx, :, :]
        extent = [0, boxsize[1], 0, boxsize[2]]
    elif losidx == 1:
        slc = cube[:, slc_idx, :]
        extent = [0, boxsize[0], 0, boxsize[2]]
    elif losidx == 2:
        slc = cube[:, :, slc_idx]
        extent = [0, boxsize[0], 0, boxsize[1]]
    else:
        raise ValueError("losidx must be 0, 1 or 2")

    fig, ax = plot_density_slice(slc, extent, **kwargs)
    return fig, ax


def plot_density_projection(cube, boxsize, losidx=2, **kwargs):
    """Plot a projection (sum along line of sight) of a 3D density cube.

    Parameters
    ----------
    cube : array_like
        3D density cube
    boxsize : array_like or float
        Box size in each dimension. If float, assumes cubic box.
    losidx : int
        Line-of-sight axis to project along (0, 1, or 2)
    **kwargs
        Additional kwargs passed to plot_density_slice

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if isinstance(boxsize, (int, float)):
        boxsize = [boxsize] * 3

    projection = np.sum(cube, axis=losidx)

    if losidx == 0:
        extent = [0, boxsize[1], 0, boxsize[2]]
    elif losidx == 1:
        extent = [0, boxsize[0], 0, boxsize[2]]
    elif losidx == 2:
        extent = [0, boxsize[0], 0, boxsize[1]]
    else:
        raise ValueError("losidx must be 0, 1 or 2")

    fig, ax = plot_density_slice(projection, extent, **kwargs)
    cbar = ax.images[0].colorbar
    cbar.set_label(r"projected density [(cMpc/h)$^{-2}$]")

    return fig, ax


def plot_density_comparison(
    fields_dict,
    boxsize,
    slc_idx=None,
    losidx=2,
    suptitle="Density Field Comparison",
    figsize=None,
    **kwargs
):
    """Plot multiple density fields side-by-side for comparison.

    Similar to jaxpm.plotting.plot_fields_single_projection.
    Useful for comparing target vs reconstructed fields.

    Parameters
    ----------
    fields_dict : dict
        Dictionary mapping field names to 3D density arrays
    boxsize : array_like or float
        Box size in each dimension. If float, assumes cubic box.
    slc_idx : int, optional
        Index of slice to plot. If None, plots middle slice.
    losidx : int
        Line-of-sight axis (0, 1, or 2) for slicing
    suptitle : str
        Overall figure title
    figsize : tuple, optional
        Figure size (width, height). If None, auto-computed based on number of fields.
    **kwargs
        Additional kwargs passed to imshow

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : array of matplotlib.axes.Axes
        Array of axes objects
    """
    if isinstance(boxsize, (int, float)):
        boxsize = [boxsize] * 3

    n_fields = len(fields_dict)
    if figsize is None:
        figsize = (5 * n_fields, 5)

    fig, axes = plt.subplots(1, n_fields, figsize=figsize)

    # Make axes iterable even if single field
    if n_fields == 1:
        axes = [axes]

    if slc_idx is None:
        slc_idx = list(fields_dict.values())[0].shape[losidx] // 2

    for ax, (name, field) in zip(axes, fields_dict.items()):
        # Extract slice
        if losidx == 0:
            slc = field[slc_idx, :, :]
            extent = [0, boxsize[1], 0, boxsize[2]]
        elif losidx == 1:
            slc = field[:, slc_idx, :]
            extent = [0, boxsize[0], 0, boxsize[2]]
        elif losidx == 2:
            slc = field[:, :, slc_idx]
            extent = [0, boxsize[0], 0, boxsize[1]]
        else:
            raise ValueError("losidx must be 0, 1 or 2")

        # Compute percentiles for consistent scaling
        pmin = np.percentile(slc, 1)
        pmin = max(0.0001, pmin)
        pmax = np.percentile(slc, 99.9)

        # Default imshow kwargs
        ikw = dict(
            cmap="viridis",
            interpolation="nearest",
            vmin=pmin,
            vmax=pmax,
            extent=extent,
            origin="lower",
            norm="log"
        )
        ikw.update(**kwargs)

        # Plot
        im = ax.imshow(slc.T, **ikw)
        ax.set_title(name)
        ax.set_xlabel("cMpc/h")
        ax.set_ylabel("cMpc/h")
        plt.colorbar(im, ax=ax, label=r"density [(cMpc/h)$^{-3}$]")

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig, axes
