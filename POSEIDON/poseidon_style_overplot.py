"""
POSEIDON-style corner plot overplot with smooth contours.

Matches the aesthetic of POSEIDON's generate_overplot output with:
- Clean contour lines (not over-filled)
- Parameter statistics on histograms
- Professional appearance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
import os

from matplotlib import rcParams
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'

# rcParams["font.family"] = "sans-serif"
# rcParams["font.sans-serif"] = ["Computer Modern Sans Serif"]
# rcParams["text.usetex"] = True
# rcParams.update({'text.latex.preamble' : r'\usepackage[version=4]{mhchem}'})

def load_multinest_samples(output_dir, model_name):
    """Load samples from POSEIDON MultiNest output files."""
    if not output_dir.endswith('/'):
        output_dir += '/'
    
    prefix = output_dir + "MultiNest_raw/" + model_name + "-"
    
    # Try equal-weighted samples first
    post_equal_file = prefix + "post_equal_weights.dat"
    if os.path.exists(post_equal_file):
        data = np.loadtxt(post_equal_file)
        samples = data[:, 0:]  # All columns are parameters
        weights = np.ones(len(samples)) / len(samples)
        return samples, weights
    
    # Try regular posterior
    post_file = prefix + ".txt"
    if os.path.exists(post_file):
        data = np.loadtxt(post_file)
        weights = data[:, 0]
        samples = data[:, 2:]
        weights = weights / np.sum(weights)
        return samples, weights
    
    raise FileNotFoundError(f"No MultiNest files found for {model_name}")


def poseidon_style_overplot(output_dir, model_names, model_params_list, 
                            params_to_plot=None,
                            labels=None,
                            model_display_names=None,
                            colors=['crimson', 'navy'],
                            figsize=None,
                            title=None,
                            save_path=None,
                            bins=40,
                            smooth=1.0,
                            contour_alpha=0.8,
                            title_fontsize=9,
                            label_fontsize=11,
                            tick_fontsize=9,
                            span=None):
    """
    Create a POSEIDON-style corner plot with clean contour lines.
    
    Parameters
    ----------
    output_dir : str
        Directory containing MultiNest_raw/ subdirectory
    model_names : list of str
        Names of models to compare
    model_params_list : list of lists
        Parameter names for each model
    params_to_plot : list of str, optional
        Which parameters to include
    labels : dict, optional
        Parameter labels for plotting
    model_display_names : list of str, optional
        Display names for legend
    colors : list of str
        Colors for each model
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    bins : int
        Number of bins for 2D histograms (default: 40)
    smooth : float
        Gaussian smoothing sigma (default: 1.0)
    contour_alpha : float
        Transparency for contour fills (default: 0.8)
    title_fontsize : int
        Font size for parameter titles (default: 9)
    label_fontsize : int
        Font size for axis labels (default: 11)
    tick_fontsize : int
        Font size for tick labels (default: 9)
    span : list of tuples, optional
        Range to plot for each parameter as [(min1, max1), (min2, max2), ...].
        If None, auto-computes from data.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    n_models = len(model_names)
    
    # Load all samples
    print("Loading samples...")
    models_data = []
    for i, (name, params) in enumerate(zip(model_names, model_params_list)):
        print(f"  Loading {name}...")
        samples, weights = load_multinest_samples(output_dir, name)
        
        models_data.append({
            'samples': samples,
            'weights': weights,
            'params': params,
            'name': name
        })
    
    # Determine parameters to plot
    if params_to_plot is None:
        all_params = []
        for params in model_params_list:
            for p in params:
                if p not in all_params:
                    all_params.append(p)
        params_to_plot = all_params
    
    n_params = len(params_to_plot)
    print(f"\nPlotting {n_params} parameters: {params_to_plot}")
    
    # Create parameter labels
    if labels is None:
        labels = {p: p for p in params_to_plot}
    
    # Create display names
    if model_display_names is None:
        model_display_names = model_names
    
    # Map samples to common parameter space
    # print("\nMapping parameters...")
    mapped_models = []
    for i, model in enumerate(models_data):
        n_samples = len(model['samples'])
        mapped_samples = np.full((n_samples, n_params), np.nan)
        
        for j, param in enumerate(model['params']):
            if param in params_to_plot:
                target_idx = params_to_plot.index(param)
                mapped_samples[:, target_idx] = model['samples'][:, j]
                # print(f"  {model_names[i]}: {param} (col {j}) -> mapped col {target_idx} (plot position for {params_to_plot[target_idx]})")
        
        mapped_models.append({
            'samples': mapped_samples,
            'weights': model['weights'],
            'name': model_display_names[i],
            'color': colors[i % len(colors)]
        })
        
        missing = [p for p in params_to_plot if p not in model['params']]
        if missing:
            print(f"  {model_names[i]} will be blank for: {missing}")
    
    # Compute or use provided ranges for each parameter
    if span is None:
        print("\nComputing parameter ranges...")
        ranges = []
        for i in range(n_params):
            all_vals = []
            for model in mapped_models:
                col_data = model['samples'][:, i]
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    all_vals.extend(valid_data)
            
            if len(all_vals) > 0:
                # Use mean +/- 5 sigma
                mean = np.mean(all_vals)
                std = np.std(all_vals)
                ranges.append((mean - 5*std, mean + 5*std))
            else:
                ranges.append((0, 1))  # Dummy range
    else:
        if len(span) != n_params:
            raise ValueError(f"span must have {n_params} tuples, got {len(span)}")
        ranges = span
        # print(f"\nUsing provided parameter ranges: {ranges}")

    
    # Create figure
    if figsize is None:
        size_per_param = 2.5
        figsize = (n_params * size_per_param, n_params * size_per_param)
    
    print(f"\nCreating figure...")
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_params, n_params, hspace=0.05, wspace=0.05)
    axes = np.empty((n_params, n_params), dtype=object)
    
    # Create plots
    for i in range(n_params):
        for j in range(n_params):
            if j > i:
                continue
            
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            
            if i == j:
                # Diagonal - 1D histogram with statistics
                stat_lines = []
                current_param = params_to_plot[i]
                
            
                for k, model in enumerate(mapped_models):
                    data = model['samples'][:, i]
                    weights = model['weights']
                    
                    # Remove NaN
                    mask = ~np.isnan(data)
                    if not np.any(mask):
                        continue
                    
                    data_clean = data[mask]
                    weights_clean = weights[mask] / weights[mask].sum()
                    
                    # Compute weighted quantiles
                    sorted_idx = np.argsort(data_clean)
                    sorted_data = data_clean[sorted_idx]
                    sorted_weights = weights_clean[sorted_idx]
                    cumsum = np.cumsum(sorted_weights)
                    
                    median = sorted_data[np.searchsorted(cumsum, 0.5)]
                    q16 = sorted_data[np.searchsorted(cumsum, 0.16)]
                    q84 = sorted_data[np.searchsorted(cumsum, 0.84)]
                    
                    upper_err = q84 - median
                    lower_err = median - q16

                    if upper_err > 10:
                        # Store with color for title
                        stat_lines.append({
                            'text': f"{median:.0f}$_{{-{lower_err:.0f}}}^{{+{upper_err:.0f}}}$",
                            'color': model['color']
                        })
                    elif upper_err > 1e-2: 
                        # Store with color for title
                        stat_lines.append({
                            'text': f"{median:.2f}$_{{-{lower_err:.2f}}}^{{+{upper_err:.2f}}}$",
                            'color': model['color']
                        })
                    else:
                        # Store with color for title
                        stat_lines.append({
                            'text': f"{median:.3f}$_{{-{lower_err:.3f}}}^{{+{upper_err:.3f}}}$",
                            'color': model['color']
                        })                        
                        
                    # Plot histogram
                    n_bins = min(50, max(15, len(data_clean) // 20))
                    hist, edges = np.histogram(data_clean, bins=n_bins, 
                                               weights=weights_clean, density=True)
                    centers = 0.5 * (edges[1:] + edges[:-1])
                    
                    # Plot filled histogram
                    label = model['name'] if i == 0 else None
                    ax.fill_between(centers, 0, hist, alpha=0.7, color=model['color'], 
                                   step='mid', label=label, linewidth=0)
                    ax.plot(centers, hist, color=model['color'], alpha=0.9, 
                           linewidth=1.8, drawstyle='steps-mid')
                    
                    # Add median line
                    ax.axvline(median, color=model['color'], linestyle='--', 
                              linewidth=2.0, alpha=0.6)
                
                # Add parameter name as title with color-coded model values
                if stat_lines:
                    # Parameter name in black - positioned ABOVE the plot box
                    param_key = params_to_plot[i]
                    param_name = labels.get(param_key, param_key) if isinstance(labels, dict) else labels[i]
                    
                    # Debug output
                    if i == 0:
                        print(f"  Title for diagonal {i}: param_key='{param_key}', param_name='{param_name}'")
                    
                    # Position above the plot box
                    y_pos = 1.00 + 0.17*n_models
                    ax.text(0.5, y_pos, param_name, transform=ax.transAxes,
                           fontsize=title_fontsize, ha='center', va='bottom',
                           color='black', weight='bold')
                    
                    # Model values in their respective colors, also above the box
                    y_pos -= 0.18  # Start below parameter name
                    for stat in stat_lines:
                        ax.text(0.5, y_pos, stat['text'], transform=ax.transAxes,
                               fontsize=title_fontsize-1, ha='center', va='bottom',
                               color=stat['color'])
                        y_pos -= 0.15  # Move down for next line
                
                ax.set_yticks([])
                if i < n_params - 1:
                    ax.set_xticks([])
                else:
                    param_key = params_to_plot[i]
                    label_text = labels.get(param_key, param_key) if isinstance(labels, dict) else labels[i]
                    ax.set_xlabel(label_text, fontsize=label_fontsize)
                    ax.tick_params(labelsize=tick_fontsize, rotation=45)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Max 4 ticks                
                # Set x limits from ranges
                ax.set_xlim(ranges[i])
                    
            else:
                # Off-diagonal - 2D contour lines
                for k, model in enumerate(mapped_models):
                    x_data = model['samples'][:, j]
                    y_data = model['samples'][:, i]
                    
                    # Remove NaN
                    mask = ~(np.isnan(x_data) | np.isnan(y_data))
                    if not np.any(mask):
                        continue
                    
                    x_clean = x_data[mask]
                    y_clean = y_data[mask]
                    w_clean = model['weights'][mask] / model['weights'][mask].sum()
                    
                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=bins, 
                                                       weights=w_clean)
                    H = H.T
                    
                    # Apply Gaussian smoothing
                    if smooth > 0:
                        H = gaussian_filter(H, sigma=smooth)
                    
                    # Normalize
                    H = H / H.sum()
                    
                    # Find contour levels (1-sigma and 2-sigma ~ 68% and 95%)
                    H_flat = H.flatten()
                    H_sorted = np.sort(H_flat)[::-1]
                    H_cumsum = np.cumsum(H_sorted)
                    
                    levels = []
                    for level_frac in [0.393, 0.865]:  # 1 and 2 sigma for 2D
                        idx = np.searchsorted(H_cumsum, level_frac)
                        if idx < len(H_sorted):
                            levels.append(H_sorted[idx])
                    
                    if len(levels) > 0:
                        levels = sorted(set(levels))
                        
                        x_centers = 0.5 * (xedges[1:] + xedges[:-1])
                        y_centers = 0.5 * (yedges[1:] + yedges[:-1])
                        
                        try:
                            # Plot very subtle filled background
                            if len(levels) >= 2:
                                # Fill between the two contour levels
                                ax.contourf(x_centers, y_centers, H, 
                                           levels=[levels[0], levels[1]],
                                           colors=[model['color']], 
                                           alpha=0.25)  # Light fill
                                # Fill inside inner contour
                                ax.contourf(x_centers, y_centers, H, 
                                           levels=[levels[1], H.max()],
                                           colors=[model['color']], 
                                           alpha=0.35)  # Slightly darker inside
                            
                            # Plot contour lines - just the edges
                            ax.contour(x_centers, y_centers, H, 
                                      levels=levels,
                                      colors=[model['color']], 
                                      linewidths=1.2,
                                      alpha=1.0,
                                      linestyles='solid')
                        except:
                            pass
                
                if j > 0:
                    ax.set_yticks([])
                else:
                    param_key = params_to_plot[i]
                    label_text = labels.get(param_key, param_key) if isinstance(labels, dict) else labels[i]
                    ax.set_ylabel(label_text, fontsize=label_fontsize)
                    ax.tick_params(axis='y', labelsize=tick_fontsize, rotation=45)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))  # Max 4 ticks                
                if i < n_params - 1:
                    ax.set_xticks([])
                else:
                    param_key = params_to_plot[j]
                    label_text = labels.get(param_key, param_key) if isinstance(labels, dict) else labels[j]
                    ax.set_xlabel(label_text, fontsize=label_fontsize)
                    ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=45)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Max 4 ticks                    
                
                # Set axis limits from ranges
                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])
    
    # Add legend to upper right of the entire figure
    if n_params > 0:
        # Create legend handles
        from matplotlib.patches import Patch
        handles = [Patch(color=model['color'], label=model['name'], alpha=0.7) 
                  for model in mapped_models]
        
        # Place legend on the figure itself, in the upper right
        fig.legend(handles=handles, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98), fontsize=18,
                  frameon=False)    
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=15, y=0.995)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        print(f"\nSaving to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Done!")
    
    return fig
