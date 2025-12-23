import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import binary_erosion


a = 0.059

def compute_free_energy(pH, U, dg0, alpha, beta):
    """Computes the dG free energy given the pH, U and parameters."""
    return dg0 - a*alpha*pH - beta*U

def plot_pourbaix(file, ax=None, output=None, N=100, exclude=None, title=None, shift=(0,0), 
                  add_contour=True, legend=False, labels=True, labelsize=8, **kwargs):
    """ 
    Plots the Pourbaix diagram given a data file.

    This function calculates the most stable surface phase across a grid of pH 
    and Potential (U) values based on Gibbs free energy. It visualizes these 
    phases as a colored map, adds Hydrogen (HER) and Oxygen (OER) evolution lines, 
    and includes smart label placement.

    Parameters
    ----------
    file : str
        Path to the data file. Expected columns: [dG0, vH, ve, label].
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    output : str, optional
        Filename/path to save the figure (e.g., 'diagram.png').
    N : int, default 1000
        Resolution of the pH and Potential grid (N x N).
    exclude : list of str, optional
        List of labels (e.g., ["F", "OH"]) to ignore during plotting.
    title : str, optional
        Title of the plot, placed in the upper right by default.
    shift : tuple of float, default (0, 0)
        Global (x, y) shift applied to all text labels.
    add_contour : bool, default True
        If True, draws black boundary lines between different stability regions.
    legend : bool, default False
        If True, adds a legend for the colored phases.
    labels : bool, default True
        If True, places text labels at the center of each stability region.
        Includes collision detection to shift labels away from HER/OER lines.
    labelsize : int, default 8
        Font size for the phase labels.
    **kwargs : dict
        Additional plotting arguments:
        - xlim (tuple): Limits for x-axis (default: (0, 14)).
        - ylim (tuple): Limits for y-axis (default: (-2, 1.5)).
        - xlabel (str): Label for x-axis (default: "pH").
        - ylabel (str): Label for y-axis (default: "$U$ (V)").
        - nlocator_x (int): Spacing for pH ticks (default: 2).
        - nlocator_y (float): Spacing for U ticks (default: 0.5).

    Returns
    -------
    plt.Axes
        The axes object containing the Pourbaix diagram.
    
    
    """

    data = pd.read_csv(file, sep='\\s+', header=None, skiprows=1,comment='#')
    if exclude is not None: data = data[~data[3].isin(exclude)]
    pla = data.iloc[:, 3].astype(str).values

    # Define pH and Potential grid
    ph  = np.linspace(*kwargs.get("xlim",(0,14)), N)
    u = np.linspace(*kwargs.get("ylim",(-2,1.5)), N)
    pH, U = np.meshgrid(ph, u)
    a = 0.059

    # Compute stability map
    stability = np.zeros_like(pH, dtype=object)
    energy_map = np.full_like(pH, np.inf, dtype=float)

    for _, row in data.iterrows():
        dg0, alpha, beta, label = row
        free_energy = compute_free_energy(pH, U, dg0, alpha, beta)
        mask = free_energy < energy_map
        energy_map[mask] = free_energy[mask]
        stability[mask] = label

    # Assign colors to each label
    unique_labels = list(set(pla))
    cmap = plt.colormaps.get_cmap("Accent")
    norm = plt.Normalize(vmin=0, vmax=len(unique_labels)-1)
    # label_to_color = {label: cmap(norm(i)) for i, label in enumerate(unique_labels)}
    label_to_color = {
        "O": "#cd6b86",
        "OH": "#eab03b",
        "H": "#6abda5",
        "F": "#24bdc5",
        "Cl": "#5d65f8",
        "Br": "#b16c51",
        "I": "#97005b",
        "S": "#e4d7df",
        "Se": "#138236",
        "clean": "lightgrey"
    }

    # Plot
    if ax is None: fig, ax = plt.subplots(figsize=(3, 3),dpi=300)
    for label in unique_labels:
        mask = stability == label
        ax.scatter(pH[mask], U[mask], color=label_to_color.get(label,"white"), label=label, s=0.5,edgecolors='none')

    # Add HER/OER lines
    h2_line = lambda p: -0.059 * p
    o2_line = lambda p: -0.059 * p + 1.23
    ax.plot(ph, h2_line(ph), 'k--', label='H2 Reduction')
    ax.plot(ph, o2_line(ph), 'k:', label='O2 Reduction')


    # Add contours
    if add_contour:
        color_map = np.zeros(pH.shape, dtype=float)
        for i, label in enumerate(unique_labels):
            mask = stability == label
            color_map[mask] = i

        boundary_mask = np.zeros_like(color_map, dtype=bool)
        for i in range(len(unique_labels)):
            phase_mask = color_map == i
            eroded_mask = binary_erosion(phase_mask)
            boundary_mask |= phase_mask ^ eroded_mask
        ax.contour(pH, U, boundary_mask, colors='black', linewidths=0.3)

    # Add text labels at region centers or arrows for small/narrow regions
    if labels:
        proximity_threshold = 0.1  # Distance in Volts to trigger shift
        label_shift_val = 0.12     # Amount to shift label by
        for label in unique_labels:
            mask = stability == label
            if np.any(mask):
                # Get the center coordinates of the region
                y_idx, x_idx = np.argwhere(mask).mean(axis=0)
                region_size = np.sum(mask)
                center_x, center_y = ph[int(x_idx)] + shift[0], u[int(y_idx)] + shift[1]

                # Check distance to HER and OER lines
                dist_her = center_y - h2_line(center_x)
                dist_oer = center_y - o2_line(center_x)

                if abs(dist_her) < proximity_threshold:
                    center_y += label_shift_val if dist_her > 0 else -label_shift_val
                if abs(dist_oer) < proximity_threshold:
                    center_y += label_shift_val if dist_oer > 0 else -label_shift_val

                if region_size/N**2 > 0.02:  # Threshold for larger regions
                    ax.text(center_x, center_y, label, ha='center', va='center', fontsize=labelsize, color='black', bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))
                else:
                    if label == "Se": center_y -= 0.15
                    arrow_x = center_x + 0.8
                    arrow_y = center_y + 0.3
                    ax.annotate(label, 
                                (center_x, center_y),  # Position of the arrowhead
                                (arrow_x, arrow_y),  # Position of the text
                                arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
                                fontsize=labelsize, color='black', 
                                bbox=dict(facecolor='none', alpha=0.5, edgecolor='none',pad=0),  ha='center', va='center')

    if title is not None: ax.set_title(title,loc='right',x=0.98,y=0.88)
    ax.set_xlabel(kwargs.get("xlabel","pH"))
    ax.set_ylabel(kwargs.get("ylabel","$U$ (V)"))
    if legend: ax.legend(markerscale=5, fontsize="xx-small",loc='upper right')
    ax.set_xlim(ph[0],ph[-1])
    ax.set_ylim(u[0],u[-1])
    ax.xaxis.set_major_locator(MultipleLocator(kwargs.get("nlocator_x",2)))
    ax.yaxis.set_major_locator(MultipleLocator(kwargs.get("nlocator_y",0.5)))
    ax.get_figure().tight_layout()
    if output is not None: plt.savefig(output, dpi=300)

    return ax



def plot_dG_vs_U(file, ax=None, pH=0, output=None, N=1000, title=None, 
                 legend=False, fill=True, labelsize=8, **kwargs):
    """
    Plot ΔG vs U for different phases at a fixed pH.

    This function visualizes the stability of different surface phases/terminations 
    by plotting ΔG as a function of U. It identifies the most stable phase 
    (the one with the lowest ΔG) for every potential value, highlights these regions 
    with color fills, and adds phase labels.
    
    Parameters
    ----------
    file : str
        Path to the data file. Expected columns: [dG0, vH, ve, label].
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    pH : float, default 0
        The fixed pH value at which to calculate free energies.
    output : str, optional
        Filename/path to save the figure (e.g., 'dg_vs_u.png').
    N : int, default 1000
        Resolution of the Potential (U) axis.
    title : str, optional
        Title of the plot, placed in the upper right.
    legend : bool, default False
        If True, adds a legend for all plotted ΔG lines.
    fill : bool, default True
        If True, shades the regions where a specific phase is most stable 
        and adds vertical separators and text labels.
    labelsize : int, default 8
        Font size for the phase labels and axis titles.
    **kwargs : dict
        Additional plotting arguments:
        - xlim (tuple): Range of Potential U (default: (-2, 1.5)).
        - ylim (tuple): Range of ΔG (default: (-15, 5)).
        - xlabel (str): Label for x-axis (default: "$U$ (V)").
        - ylabel (str): Label for y-axis (default: "Δ$G$ (eV)").

    Returns
    -------
    plt.Axes
        The axes object containing the ΔG vs U plot.
    
    """
    xlim,ylim = kwargs.get("xlim",(-2,1.5)),kwargs.get("ylim",(-15,5))

    # Load data
    data = pd.read_csv(file, sep=r'\s+', header=None, skiprows=1, comment='#')
    data.columns = ["dG0", "alpha", "beta", "label"]
    labels = data["label"].astype(str).values

    # Potential axis
    U = np.linspace(*xlim, N)

    # Compute free energy curves
    free_energies = {}
    for _, row in data.iterrows():
        dg0, alpha, beta, label = row
        free_energies[label] = compute_free_energy(pH, U, dg0, alpha, beta)

    # Find lowest free energy phase at each U
    all_dG = np.vstack(list(free_energies.values()))
    min_idx = np.argmin(all_dG, axis=0)
    unique_labels = list(free_energies.keys())

    # Custom palette for some terminations
    custom_palette = {
        "O": "#cd6b86",
        "OH": "#eab03b",
        "H": "#6abda5",
        "F": "#24bdc5",
        "Cl": "#5d65f8",
        "Br": "#b16c51",
        "Br": "#b16c51",
        "I": "#97005b",
        "S": "#d7bfd1",
        "Se": "#138236",
        "clean": "lightgrey"
    }

    # Colormap for remaining labels
    cmap = plt.colormaps.get_cmap("tab10")
    remaining_labels = [lab for lab in unique_labels if lab not in custom_palette]
    remaining_colors = {lab: cmap(i / max(1, len(remaining_labels)-1)) 
                        for i, lab in enumerate(remaining_labels)}

    # Merge palettes
    label_to_color = {**custom_palette, **remaining_colors}

    # Setup plot
    if ax is None: fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    # Plot all ΔG(U) lines
    for label, dG in free_energies.items():
        ax.plot(U, dG, label=label, color=label_to_color[label], lw=1)

    # Highlight most stable regions
    if fill:
        stable_labels = np.array(unique_labels)[min_idx]
        prev_label = stable_labels[0]
        start_idx = 0
        y_bottom = ylim[0] if ylim else None  # baseline for fill

        for i in range(1, len(U)):
            if stable_labels[i] != prev_label or i == len(U)-1:
                end_idx = i
                baseline = y_bottom if y_bottom is not None else ax.get_ylim()[0]

                # Fill region
                ax.fill_between(U[start_idx:end_idx],
                                free_energies[prev_label][start_idx:end_idx],
                                baseline,
                                color=label_to_color[prev_label], alpha=0.3)

                # Vertical separator
                ax.axvline(U[end_idx], color="k", ls="--", lw=0.5)

                # Add label at bottom center of region
                U_mid = (U[start_idx] + U[end_idx]) / 2
                ax.text(U_mid, baseline+0.5, prev_label,
                        ha="center", va="bottom",
                        fontsize=labelsize, color="black",
                        bbox=dict(facecolor="none", edgecolor="none", alpha=0.7, pad=0.2))

                prev_label = stable_labels[i]
                start_idx = end_idx

    # Labels, limits, etc.
    if title: ax.set_title(title, loc="right", x=0.98, y=0.89)
    ax.set_xlabel(kwargs.get("xlabel", "$U$ (V)"))
    ax.set_ylabel(kwargs.get("ylabel", "Δ$G$ (eV)"))
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    if legend:
        ax.legend(fontsize="x-small", loc="best")
    ax.get_figure().tight_layout()


    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches="tight")

    return ax

def plot_dG_vs_pH(file, U=0, ax=None, output=None,  N=1000, title=None, 
                  legend=False, fill=True, labelsize=8, **kwargs):
    """
    Plot ΔG vs pH for different phases at a fixed potential U.

    This function visualizes the stability of different surface phases/terminations 
    by plotting ΔG as a function of pH. It identifies the most stable phase 
    (the one with the lowest ΔG) for every potential value, highlights these regions 
    with color fills, and adds phase labels.
    
    Parameters
    ----------
    file : str
        Path to the data file. Expected columns: [dG0, vH, ve, label].
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    U  : float, default 0
        The fixed U value at which to calculate free energies.
    output : str, optional
        Filename/path to save the figure (e.g., 'dg_vs_ph.png').
    N : int, default 1000
        Resolution of the pH axis.
    title : str, optional
        Title of the plot, placed in the upper right.
    legend : bool, default False
        If True, adds a legend for all plotted ΔG lines.
    fill : bool, default True
        If True, shades the regions where a specific phase is most stable 
        and adds vertical separators and text labels.
    labelsize : int, default 8
        Font size for the phase labels and axis titles.
    **kwargs : dict
        Additional plotting arguments:
        - xlim (tuple): Range of pH values (default: (0, 15)).
        - ylim (tuple): Range of ΔG (default: (-10, 2)).
        - xlabel (str): Label for x-axis (default: "pH").
        - ylabel (str): Label for y-axis (default: "Δ$G$ (eV)").

    Returns
    -------
    plt.Axes
        The axes object containing the ΔG vs U plot.    """
    xlim,ylim = kwargs.get("xlim",(0,14)),kwargs.get("ylim",(-10,2))

    # Load data
    data = pd.read_csv(file, sep=r'\s+', header=None, skiprows=1, comment='#')
    data.columns = ["dG0", "alpha", "beta", "label"]
    labels = data["label"].astype(str).values

    # pH axis
    pH_vals = np.linspace(*xlim, N)

    # Compute free energy curves
    free_energies = {}
    for _, row in data.iterrows():
        dg0, alpha, beta, label = row
        free_energies[label] = compute_free_energy(pH_vals, U, dg0, alpha, beta)

    # Find lowest free energy phase at each pH
    all_dG = np.vstack(list(free_energies.values()))
    min_idx = np.argmin(all_dG, axis=0)
    unique_labels = list(free_energies.keys())

    # Custom palette for some terminations
    custom_palette = {
        "O": "#cd6b86",
        "OH": "#eab03b",
        "H": "#6abda5",
        "F": "#24bdc5",
        "Cl": "#5d65f8",
        "Br": "#b16c51",
        "I": "#97005b",
        "S": "#d7bfd1",
        "Se": "#138236",
        "clean": "lightgrey"
    }

    # Colormap for remaining labels
    cmap = plt.colormaps.get_cmap("tab10")
    remaining_labels = [lab for lab in unique_labels if lab not in custom_palette]
    remaining_colors = {lab: cmap(i / max(1, len(remaining_labels)-1)) 
                        for i, lab in enumerate(remaining_labels)}

    # Merge palettes
    label_to_color = {**custom_palette, **remaining_colors}

    # Setup plot
    if ax is None: fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    # Plot all ΔG(pH) lines
    for label, dG in free_energies.items():
        ax.plot(pH_vals, dG, label=label, color=label_to_color[label], lw=1)

    # Highlight most stable regions
    if fill:
        stable_labels = np.array(unique_labels)[min_idx]
        prev_label = stable_labels[0]
        start_idx = 0
        y_bottom = ylim[0] if ylim else None  # baseline for fill

        for i in range(1, len(pH_vals)):
            if stable_labels[i] != prev_label or i == len(pH_vals)-1:
                end_idx = i
                baseline = y_bottom if y_bottom is not None else ax.get_ylim()[0]

                # Fill region
                ax.fill_between(pH_vals[start_idx:end_idx],
                                free_energies[prev_label][start_idx:end_idx],
                                baseline,
                                color=label_to_color[prev_label], alpha=0.3)

                # Vertical separator
                ax.axvline(pH_vals[end_idx], color="k", ls="--", lw=0.5)

                # Add label at bottom center of region
                pH_mid = (pH_vals[start_idx] + pH_vals[end_idx]) / 2
                ax.text(pH_mid, baseline + 0.5, prev_label,
                        ha="center", va="bottom",
                        fontsize=labelsize, color="black",
                        bbox=dict(facecolor="none", edgecolor="none", alpha=0.7, pad=0.2))

                prev_label = stable_labels[i]
                start_idx = end_idx

    # Labels, limits, etc.
    if title: ax.set_title(title, loc="right", x=0.98, y=0.89)
    ax.set_xlabel(kwargs.get("xlabel", "pH"))
    ax.set_ylabel(kwargs.get("ylabel", "Δ$G$ (eV)"))
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    if legend:
        ax.legend(fontsize="x-small", loc="best")
    ax.get_figure().tight_layout()


    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches="tight")

    return ax

