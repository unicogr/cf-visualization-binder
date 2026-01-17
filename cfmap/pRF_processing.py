import neuropythy as ny  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Wedge

def plot_prf_maps(prf_results, flatmaps, colors_ecc, colors_polar, h, max_sigma=3, r2_threshold=0.15):
    """Plots pRF parameter maps (eccentricity, polar angle, sigma, and R²) on a cortical flatmap.

    Parameters
    ----------
    prf_results : pd.DataFrame
        DataFrame containing pRF parameters with columns 'x', 'y', 'sd', 'r2'.
    flatmaps : dict
        Dictionary of flatmaps for hemispheres, e.g., flatmaps[h].
    colors_ecc : dict
        Color palette for eccentricity with 'matplotlib_cmap' and 'hex' keys.
    colors_polar : dict
        Color palette for polar angle with 'matplotlib_cmap' and 'hex' keys.
    h : str
        Hemisphere identifier, e.g., 'lh' or 'rh'.
    max_sigma : float, optional
        Maximum sigma value for colorbar scaling (default: 3).
    r2_threshold : float, optional
        R² threshold for alpha scaling (default: 0.15).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    # Calculate derived measures from x and y coordinates
    x = prf_results['x'].values
    y = prf_results['y'].values  # Flip y coordinates as in original code
    ecc = np.abs(x + 1j*y)  # Eccentricity
    polar = np.angle(x + 1j*y)  # Polar angle
    sigma = prf_results['sd'].values  # pRF size
    r2 = prf_results['r2'].values  # Variance explained

    # Create alpha based on R²
    alpha = np.clip(r2 / 0.5, 0, 1)  # Scale alpha by R², capped at 0.5 R² for max visibility

    # Create the maps
    ecc_map = ecc
    polar_map = polar
    sigma_map = sigma
    r2_map = r2

    # Create the figure
    fig, (left_ax, left_middle_ax, right_middle_ax, right_ax) = plt.subplots(1, 4, figsize=(16, 4), dpi=72*4)

    # Eccentricity plot
    ny.cortex_plot(
        flatmaps[h],
        axes=left_ax,
        color=ecc_map,
        cmap=colors_ecc['matplotlib_cmap'],
        vmin=np.min(ecc_map),
        vmax=np.max(ecc_map)/4,
        alpha=alpha
    )

    # Polar angle plot
    ny.cortex_plot(
        flatmaps[h],
        axes=left_middle_ax,
        color=polar_map,
        cmap=colors_polar['matplotlib_cmap'],
        alpha=alpha
    )

    # Sigma (pRF size) plot
    size_vmin = np.min(sigma_map)
    size_vmax = np.max(sigma_map)
    print(f"Sigma range: {size_vmin:.2f} - {size_vmax:.2f}")

    size_cmap = plt.cm.jet
    ny.cortex_plot(
        flatmaps[h],
        axes=right_middle_ax,
        color=sigma_map,
        cmap=size_cmap,
        vmin=np.min(sigma_map),
        vmax=max_sigma,
        alpha=alpha
    )

    # Variance explained plot
    varex_cmap = plt.cm.inferno
    ny.cortex_plot(
        flatmaps[h],
        axes=right_ax,
        color=r2_map,
        cmap=varex_cmap,
        vmin=0,
        vmax=1
    )

    # Add legends/inserts

    # Eccentricity inset
    ecc_inset = inset_axes(
        left_ax,
        width="50%",
        height="50%",
        loc="lower right",
        borderpad=-6
    )
    ecc_inset.set_aspect('equal')
    ecc_inset.set_xlim(-1.5, 1.5)
    ecc_inset.set_ylim(-1.5, 1.5)
    ecc_inset.text(0.5, -0.05, r'$\mathit{r}\ (\mathit{deg})$', ha='center', va='top', fontsize=14, transform=ecc_inset.transAxes)
    ecc_inset.set_axis_off()

    # Add concentric rings for eccentricity
    num_ecc_colors = len(colors_ecc["hex"])
    for i, color in enumerate(colors_ecc["hex"]):
        inner_r = i / num_ecc_colors
        outer_r = (i + 1) / num_ecc_colors
        ring = Wedge((0, 0), outer_r, 0, 360,
                     width=outer_r - inner_r,
                     color=color)
        ecc_inset.add_patch(ring)

    # Polar angle inset
    polar_inset = inset_axes(
        left_middle_ax,
        width="40%",
        height="40%",
        loc="lower right",
        borderpad=-6
    )
    polar_inset.set_aspect('equal')
    polar_inset.set_axis_off()
    polar_inset.pie(
        [1]*len(colors_polar["hex"]),
        colors=colors_polar["hex"],
        startangle=180,
        counterclock=False
    )
    polar_inset.text(0.5, -0.05, r'$\theta\ (\mathit{rad})$', ha='center', va='top', fontsize=14, transform=polar_inset.transAxes)

    # Sigma colorbar
    sigma_rect_ax = inset_axes(
        right_middle_ax,
        width="30%",
        height="10%",
        loc="lower right", 
        borderpad=-3
    )
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    sigma_rect_ax.imshow(gradient, aspect='auto', cmap=size_cmap, extent=[0, 1, 0, 1])
    sigma_rect_ax.text(0, -0.3, f'{np.min(sigma_map):.2f}', ha='left', va='top', fontsize=10)
    sigma_rect_ax.text(1, -0.3, f'{max_sigma:.2f}', ha='right', va='top', fontsize=10)
    sigma_rect_ax.text(0.5, 1.3, r'$\sigma\ (\mathit{deg})$', ha='center', va='bottom', fontsize=14, transform=sigma_rect_ax.transAxes)
    sigma_rect_ax.axis('off')

    # Variance explained colorbar
    varex_rect_ax = inset_axes(
        right_ax,
        width="30%",
        height="10%",
        loc="lower right",          
        borderpad=-3
    )
    varex_rect_ax.imshow(gradient, aspect='auto', cmap=varex_cmap, extent=[0, 1, 0, 1])
    varex_rect_ax.text(0, -0.3, '0', ha='left', va='top', fontsize=12)
    varex_rect_ax.text(1, -0.3, '1', ha='right', va='top', fontsize=12)
    varex_rect_ax.text(0.5, 1.3, r'$\mathit{r}\!{}^2$', ha='center', va='bottom', fontsize=14, transform=varex_rect_ax.transAxes)
    varex_rect_ax.axis('off')

    # Clean up axes
    left_ax.axis('off')
    left_middle_ax.axis('off')
    right_middle_ax.axis('off')
    right_ax.axis('off')

    plt.tight_layout()
    return fig


def plot_prf_histograms(x, y, sigma, total_rsq, r2_threshold=0.0, ylims=None, xlims=None, bins=200):
    """Plot histograms of pRF parameters.
    
    Parameters
    ----------
    x : array_like
        X coordinates of pRF centers.
    y : array_like
        Y coordinates of pRF centers.
    sigma : array_like
        pRF size values.
    total_rsq : array_like
        R² values.
    r2_threshold : float, optional
        Minimum R² to include (default: 0.0).
    ylims : list of tuples, optional
        List of 4 tuples for y-axis limits [(y0_min, y0_max), (y1_min, y1_max), ...].
    xlims : list of tuples, optional
        List of 4 tuples for x-axis limits [(x0_min, x0_max), (x1_min, x1_max), ...].
    bins : int or list, optional
        Number of bins or list of 4 bin specifications (default: 200).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    # Calculate eccentricity and polar angle
    ecc = np.abs(x + 1j*y)
    polar = np.angle(x + 1j*y)
    
    # Create mask for significant voxels
    mask = total_rsq > r2_threshold
    
    # Handle bins parameter - can be single value or list of 4
    if isinstance(bins, (int, str)):
        bins_list = [bins] * 4
    else:
        bins_list = bins
    
    fig, axes = plt.subplots(1, 4, figsize=(9, 3), dpi=200)
    params = [
        (ecc[mask], r'$\mathit{r}\ (\mathit{deg})$', 'black', 
         None if ylims is None else ylims[0],
         None if xlims is None else xlims[0],
         bins_list[0]),
        (polar[mask], r'$\theta\ (\mathit{rad})$', 'black', 
         None if ylims is None else ylims[1],
         None if xlims is None else xlims[1],
         bins_list[1]),
        (sigma[mask], r'$\sigma\ (\mathit{deg})$', 'black', 
         None if ylims is None else ylims[2],
         None if xlims is None else xlims[2],
         bins_list[2]),
        (total_rsq[mask], r'$r^2$', 'black', 
         None if ylims is None else ylims[3],
         None if xlims is None else xlims[3],
         bins_list[3])
    ]
    
    for ax, (data, xlabel, color, ylim, xlim, n_bins) in zip(axes, params):
        ax.hist(data, bins=n_bins, color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Set x-axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(np.min(data), np.max(data))
        
        # Set y-axis limits
        if ylim is not None:
            ax.set_ylim(ylim)
        
        ax.set_title(xlabel, fontsize=16)
    
    # Add text showing the number of vertices
    total_vertices = len(total_rsq)
    significant_vertices = np.sum(mask)
    fig.text(0.01, -0.1, 
             f'Total vertices: {total_vertices}\nSignificant (R²>{r2_threshold:.2f}): {significant_vertices} ({significant_vertices/total_vertices*100:.1f}%)', 
             fontsize=8, ha='left')
    
    plt.tight_layout()
    return fig


def plot_roi_retMap(ecc_map, pol_map, mask, titles, flatmaps, colors_ecc, colors_polar, h):
    """Plots eccentricity and polar angle maps on a cortical flatmap.

    Parameters
    ----------
    ecc_map : array_like
        Eccentricity values for each vertex.
    pol_map : array_like
        Polar angle values for each vertex.
    mask : array_like
        Boolean mask to apply to the plots.
    titles : list of str
        List containing two title strings for eccentricity and polar angle plots.
    flatmaps : dict
        Dictionary of flatmaps for hemispheres.
    colors_ecc : dict
        Color palette for eccentricity.
    colors_polar : dict
        Color palette for polar angle.
    h : str
        Hemisphere identifier (e.g., 'lh' or 'rh').
    """
    # Create figure with two subplots
    fig, (ax_ecc, ax_polar) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    # Eccentricity plot
    ny.cortex_plot(
        flatmaps[h],
        axes=ax_ecc,
        color=ecc_map,
        cmap=colors_ecc['matplotlib_cmap'],
        mask=mask,
        vmin=np.min(ecc_map[mask]),
        vmax=np.max(ecc_map[mask]) / 2
    )

    # Polar angle plot
    ny.cortex_plot(
        flatmaps[h],
        axes=ax_polar,
        color=pol_map,
        cmap=colors_polar['matplotlib_cmap'],
        mask=mask
    )

    # Set aspect ratio and titles
    ax_ecc.set_aspect('equal')
    ax_polar.set_aspect('equal')
    ax_ecc.set_title(titles[0], fontsize=16)
    ax_polar.set_title(titles[1], fontsize=16)

    # Eccentricity inset (concentric rings)
    ecc_inset = inset_axes(
        ax_ecc,
        width="50%",
        height="50%",
        loc="lower right",
        borderpad=-6
    )
    ecc_inset.set_aspect('equal')
    ecc_inset.set_xlim(-1.5, 1.5)
    ecc_inset.set_ylim(-1.5, 1.5)
    ecc_inset.text(0.5, -0.05, r'$\rho\ (\mathit{deg})$', ha='center', va='top', fontsize=14, transform=ecc_inset.transAxes)
    ecc_inset.set_axis_off()

    # Create concentric rings for eccentricity
    num_ecc_colors = len(colors_ecc["hex"])
    for i, color in enumerate(colors_ecc["hex"]):
        inner_r = i / num_ecc_colors
        outer_r = (i + 1) / num_ecc_colors
        ring = Wedge((0, 0), outer_r, 0, 360,
                     width=outer_r - inner_r,
                     color=color)
        ecc_inset.add_patch(ring)

    # Polar angle inset (pie)
    polar_inset = inset_axes(
        ax_polar,
        width="40%",
        height="40%",
        loc="lower right",
        borderpad=-6
    )
    polar_inset.set_aspect('equal')
    polar_inset.set_axis_off()
    polar_inset.pie(
        [1]*len(colors_polar["hex"]),
        colors=colors_polar["hex"],
        startangle=180,
        counterclock=False
    )
    polar_inset.text(0.5, -0.05, r'$\theta\ (\mathit{rad})$', ha='center', va='top', fontsize=14, transform=polar_inset.transAxes)

    # Turn off axes
    ax_ecc.axis('off')
    ax_polar.axis('off')

    plt.tight_layout()
    plt.show()