"""Plotting utilities for density field visualization."""

import numpy as np
import matplotlib.pyplot as plt


def density_to_overdensity(density):
    """Convert density field to overdensity δ = ρ/ρ̄ - 1.

    Parameters
    ----------
    density : array_like
        Density field

    Returns
    -------
    overdensity : array_like
        Overdensity field (δ = ρ/ρ̄ - 1)
    """
    mean_density = np.mean(density)
    return density / mean_density - 1.0


def plot_overdensity_slice(slc, extent, imshow_kwargs=None):
    """Plot a 2D slice of an overdensity field.

    Parameters
    ----------
    slc : array_like
        2D overdensity slice to plot
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

    # Use symmetric limits around 0 for overdensity
    vmax = np.percentile(np.abs(slc), 99)
    vmin = -vmax

    if imshow_kwargs is None:
        imshow_kwargs = {}

    ikw = dict(
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
    )
    ikw.update(**imshow_kwargs)

    ax.imshow(slc.T, **ikw)
    ax.set_xlabel("cMpc/h")
    ax.set_ylabel("cMpc/h")
    cbar = plt.colorbar(ax.images[0], ax=ax)
    cbar.set_label(r"overdensity $\delta$")

    return fig, ax


def plot_overdensity_comparison(
    density_init,
    density_final,
    boxsize,
    a_init,
    a_final,
    slc_idx=None,
    losidx=2,
    figsize=(12, 5),
):
    """Plot initial and final overdensity fields side-by-side.

    Parameters
    ----------
    density_init : array_like
        Initial 3D density field
    density_final : array_like
        Final 3D density field
    boxsize : float
        Box size in Mpc/h
    a_init : float
        Initial scale factor
    a_final : float
        Final scale factor
    slc_idx : int, optional
        Index of slice to plot. If None, plots middle slice.
    losidx : int
        Line-of-sight axis (0, 1, or 2) for slicing
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : tuple of matplotlib.axes.Axes
        Tuple of (ax_init, ax_final)
    """
    if isinstance(boxsize, (int, float)):
        boxsize = [boxsize] * 3

    # Convert to overdensity
    delta_init = density_to_overdensity(density_init)
    delta_final = density_to_overdensity(density_final)

    if slc_idx is None:
        slc_idx = density_init.shape[losidx] // 2

    # Extract slices
    if losidx == 0:
        slc_init = delta_init[slc_idx, :, :]
        slc_final = delta_final[slc_idx, :, :]
        extent = [0, boxsize[1], 0, boxsize[2]]
    elif losidx == 1:
        slc_init = delta_init[:, slc_idx, :]
        slc_final = delta_final[:, slc_idx, :]
        extent = [0, boxsize[0], 0, boxsize[2]]
    elif losidx == 2:
        slc_init = delta_init[:, :, slc_idx]
        slc_final = delta_final[:, :, slc_idx]
        extent = [0, boxsize[0], 0, boxsize[1]]
    else:
        raise ValueError("losidx must be 0, 1 or 2")

    # Use same color scale for both panels (based on final which has more contrast)
    vmax = np.percentile(np.abs(slc_final), 99)
    vmin = -vmax

    fig, (ax_init, ax_final) = plt.subplots(1, 2, figsize=figsize)

    ikw = dict(
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
    )

    # Initial panel
    im_init = ax_init.imshow(slc_init.T, **ikw)
    ax_init.set_xlabel("cMpc/h")
    ax_init.set_ylabel("cMpc/h")
    ax_init.text(
        0.05, 0.95, f"a = {a_init:.2f}",
        transform=ax_init.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Final panel
    im_final = ax_final.imshow(slc_final.T, **ikw)
    ax_final.set_xlabel("cMpc/h")
    ax_final.set_ylabel("cMpc/h")
    ax_final.text(
        0.05, 0.95, f"a = {a_final:.2f}",
        transform=ax_final.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Single colorbar for both panels
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_final, cax=cbar_ax)
    cbar.set_label(r"overdensity $\delta$")

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    return fig, (ax_init, ax_final)


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


def plot_reconstruction_comparison_2x2(
    target_density_init,
    target_density_final,
    reconstructed_density_init,
    reconstructed_density_final,
    boxsize,
    a_init,
    a_final,
    slc_idx=None,
    losidx=2,
    figsize=(12, 12),
):
    """Plot 2x2 comparison of target vs reconstructed density evolution.

    Top row: Target (initial left, final right)
    Bottom row: Reconstructed (initial left, final right)

    Parameters
    ----------
    target_density_init : array_like
        Target initial 3D density field
    target_density_final : array_like
        Target final 3D density field
    reconstructed_density_init : array_like
        Reconstructed initial 3D density field
    reconstructed_density_final : array_like
        Reconstructed final 3D density field
    boxsize : float
        Box size in Mpc/h
    a_init : float
        Initial scale factor
    a_final : float
        Final scale factor
    slc_idx : int, optional
        Index of slice to plot. If None, plots middle slice.
    losidx : int
        Line-of-sight axis (0, 1, or 2) for slicing
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : array of matplotlib.axes.Axes
        2x2 array of axes
    """
    if isinstance(boxsize, (int, float)):
        boxsize = [boxsize] * 3

    # Convert to overdensity
    target_delta_init = density_to_overdensity(target_density_init)
    target_delta_final = density_to_overdensity(target_density_final)
    recon_delta_init = density_to_overdensity(reconstructed_density_init)
    recon_delta_final = density_to_overdensity(reconstructed_density_final)

    if slc_idx is None:
        slc_idx = target_density_init.shape[losidx] // 2

    # Extract slices
    if losidx == 0:
        target_slc_init = target_delta_init[slc_idx, :, :]
        target_slc_final = target_delta_final[slc_idx, :, :]
        recon_slc_init = recon_delta_init[slc_idx, :, :]
        recon_slc_final = recon_delta_final[slc_idx, :, :]
        extent = [0, boxsize[1], 0, boxsize[2]]
    elif losidx == 1:
        target_slc_init = target_delta_init[:, slc_idx, :]
        target_slc_final = target_delta_final[:, slc_idx, :]
        recon_slc_init = recon_delta_init[:, slc_idx, :]
        recon_slc_final = recon_delta_final[:, slc_idx, :]
        extent = [0, boxsize[0], 0, boxsize[2]]
    elif losidx == 2:
        target_slc_init = target_delta_init[:, :, slc_idx]
        target_slc_final = target_delta_final[:, :, slc_idx]
        recon_slc_init = recon_delta_init[:, :, slc_idx]
        recon_slc_final = recon_delta_final[:, :, slc_idx]
        extent = [0, boxsize[0], 0, boxsize[1]]
    else:
        raise ValueError("losidx must be 0, 1 or 2")

    # Compute color scales
    # Initial: 30x smaller range than final
    vmax_final = np.percentile(np.abs(target_slc_final), 99)
    vmax_init = vmax_final / 30.0
    vmin_final = -vmax_final
    vmin_init = -vmax_init

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ikw_init = dict(
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=vmin_init,
        vmax=vmax_init,
        extent=extent,
        origin="lower",
    )

    ikw_final = dict(
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=vmin_final,
        vmax=vmax_final,
        extent=extent,
        origin="lower",
    )

    # Top-left: Target initial
    im_target_init = axes[0, 0].imshow(target_slc_init.T, **ikw_init)
    axes[0, 0].set_title(f"Target Initial (a = {a_init:.2f})", fontsize=12)
    axes[0, 0].set_xlabel("cMpc/h")
    axes[0, 0].set_ylabel("cMpc/h")

    # Top-right: Target final
    im_target_final = axes[0, 1].imshow(target_slc_final.T, **ikw_final)
    axes[0, 1].set_title(f"Target Final (a = {a_final:.2f})", fontsize=12)
    axes[0, 1].set_xlabel("cMpc/h")
    axes[0, 1].set_ylabel("cMpc/h")

    # Bottom-left: Reconstructed initial
    im_recon_init = axes[1, 0].imshow(recon_slc_init.T, **ikw_init)
    axes[1, 0].set_title(f"Reconstructed Initial (a = {a_init:.2f})", fontsize=12)
    axes[1, 0].set_xlabel("cMpc/h")
    axes[1, 0].set_ylabel("cMpc/h")

    # Bottom-right: Reconstructed final
    im_recon_final = axes[1, 1].imshow(recon_slc_final.T, **ikw_final)
    axes[1, 1].set_title(f"Reconstructed Final (a = {a_final:.2f})", fontsize=12)
    axes[1, 1].set_xlabel("cMpc/h")
    axes[1, 1].set_ylabel("cMpc/h")

    # Add colorbars
    # Adjust figure to make room for colorbars on both sides
    fig.subplots_adjust(left=0.12, right=0.88)

    # Colorbar for initial densities (LEFT side for left column)
    cbar_ax_init = fig.add_axes([0.02, 0.53, 0.02, 0.35])
    cbar_init = fig.colorbar(im_target_init, cax=cbar_ax_init)
    cbar_init.set_label(r"Initial $\delta$ (30× zoom)", fontsize=10)

    # Colorbar for final densities (RIGHT side for right column)
    cbar_ax_final = fig.add_axes([0.92, 0.53, 0.02, 0.35])
    cbar_final = fig.colorbar(im_target_final, cax=cbar_ax_final)
    cbar_final.set_label(r"Final $\delta$", fontsize=10)

    plt.tight_layout(rect=[0.12, 0, 0.88, 1])

    return fig, axes


def generate_reconstruction_gif(
    iterations_ics,
    target_density_init,
    target_density_final,
    model,
    boxsize,
    a_init,
    a_final,
    output_path,
    rtol=1e-2,
    atol=1e-2,
    fps=2,
    slice_depth=None
):
    """
    Generate GIF animation of reconstruction optimization iterations.

    Creates a 2x2 layout where:
    - Top row (target): remains static
    - Bottom row (reconstructed): updates each iteration

    Parameters
    ----------
    iterations_ics : list of (int, array)
        List of (iteration_number, initial_conditions) tuples from optimization
    target_density_init : array_like
        Target initial density field (static)
    target_density_final : array_like
        Target final density field (static)
    model : Davis1985Simulation
        Simulation model to run forward simulations
    boxsize : float
        Box size in cMpc/h
    a_init : float
        Initial scale factor
    a_final : float
        Final scale factor
    output_path : str
        Path to save the GIF file
    rtol : float
        Relative tolerance for ODE integrator
    atol : float
        Absolute tolerance for ODE integrator
    fps : int
        Frames per second for GIF
    slice_depth : int, optional
        Depth for slice (default: middle of box)

    Returns
    -------
    str
        Path to generated GIF file
    """
    import os
    from pathlib import Path

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow required for GIF generation. Install with: uv add pillow")

    print(f"\nGenerating GIF with {len(iterations_ics)} frames...")
    print(f"Output: {output_path}")

    # Create frames subdirectory alongside the GIF
    output_path_obj = Path(output_path)
    frames_dir = output_path_obj.parent / "frames"
    frames_dir.mkdir(exist_ok=True, parents=True)
    frame_paths = []

    print(f"Saving frames to: {frames_dir}")

    # Determine slice depth
    if slice_depth is None:
        slice_depth = target_density_init.shape[2] // 2

    # Convert target to overdensity (static)
    target_delta_init = density_to_overdensity(target_density_init)
    target_delta_final = density_to_overdensity(target_density_final)

    # Get slices for static target
    target_slc_init = target_delta_init[:, :, slice_depth]
    target_slc_final = target_delta_final[:, :, slice_depth]

    # Compute color scales
    vmax_final = np.percentile(np.abs(target_slc_final), 99)
    vmax_init = vmax_final / 30.0

    extent = [0, boxsize, 0, boxsize]

    for idx, (iteration, ics) in enumerate(iterations_ics):
        print(f"  Frame {idx+1}/{len(iterations_ics)}: iteration {iteration}", flush=True)

        # Run forward simulation with current ICs
        positions, _ = model.run_simulation(ics, rtol=rtol, atol=atol)

        # Paint densities
        recon_density_init = model.paint_density(positions[0])
        recon_density_final = model.paint_density(positions[-1])

        # Convert to overdensity
        recon_delta_init = density_to_overdensity(recon_density_init)
        recon_delta_final = density_to_overdensity(recon_density_final)

        # Get slices
        recon_slc_init = recon_delta_init[:, :, slice_depth]
        recon_slc_final = recon_delta_final[:, :, slice_depth]

        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Imshow kwargs
        ikw_init = {'extent': extent, 'origin': 'lower', 'cmap': 'RdBu_r',
                   'vmin': -vmax_init, 'vmax': vmax_init, 'aspect': 'auto'}
        ikw_final = {'extent': extent, 'origin': 'lower', 'cmap': 'RdBu_r',
                    'vmin': -vmax_final, 'vmax': vmax_final, 'aspect': 'auto'}

        # Top-left: Target initial (static)
        im_target_init = axes[0, 0].imshow(target_slc_init.T, **ikw_init)
        axes[0, 0].set_title(f"Target Initial (a = {a_init:.2f})", fontsize=12)
        axes[0, 0].set_xlabel("cMpc/h")
        axes[0, 0].set_ylabel("cMpc/h")

        # Top-right: Target final (static)
        im_target_final = axes[0, 1].imshow(target_slc_final.T, **ikw_final)
        axes[0, 1].set_title(f"Target Final (a = {a_final:.2f})", fontsize=12)
        axes[0, 1].set_xlabel("cMpc/h")
        axes[0, 1].set_ylabel("cMpc/h")

        # Bottom-left: Reconstructed initial (updating)
        im_recon_init = axes[1, 0].imshow(recon_slc_init.T, **ikw_init)
        axes[1, 0].set_title(f"Reconstructed Initial (iter {iteration})", fontsize=12)
        axes[1, 0].set_xlabel("cMpc/h")
        axes[1, 0].set_ylabel("cMpc/h")

        # Bottom-right: Reconstructed final (updating)
        im_recon_final = axes[1, 1].imshow(recon_slc_final.T, **ikw_final)
        axes[1, 1].set_title(f"Reconstructed Final (iter {iteration})", fontsize=12)
        axes[1, 1].set_xlabel("cMpc/h")
        axes[1, 1].set_ylabel("cMpc/h")

        # Add colorbars
        fig.subplots_adjust(left=0.12, right=0.88)

        # LEFT colorbar for initial densities
        cbar_ax_init = fig.add_axes([0.02, 0.53, 0.02, 0.35])
        cbar_init = fig.colorbar(im_target_init, cax=cbar_ax_init)
        cbar_init.set_label(r"Initial $\delta$ (30× zoom)", fontsize=10)

        # RIGHT colorbar for final densities
        cbar_ax_final = fig.add_axes([0.92, 0.53, 0.02, 0.35])
        cbar_final = fig.colorbar(im_target_final, cax=cbar_ax_final)
        cbar_final.set_label(r"Final $\delta$", fontsize=10)

        plt.tight_layout(rect=[0.12, 0, 0.88, 1])

        # Save frame to permanent directory
        frame_filename = f"frame_{idx:04d}_iter_{iteration:04d}.png"
        frame_path = frames_dir / frame_filename
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        frame_paths.append(str(frame_path))
        plt.close(fig)

    # Create GIF from frames
    print(f"\nCompiling {len(frame_paths)} frames into GIF...")
    frames = [Image.open(fp) for fp in frame_paths]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    print(f"GIF saved to: {output_path}")
    print(f"Individual frames saved to: {frames_dir}")

    return output_path
