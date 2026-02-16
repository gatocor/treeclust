"""
Multiresolution dotplot visualization for treeclust.

This module provides functions for creating dotplots/scatterplots showing
cluster expression profiles across different resolutions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings

def plot_cluster_expression_dotplot(
    expression_data: pd.DataFrame,
    clustering_results: Dict[float, np.ndarray],
    properties: Optional[List[str]] = None,
    resolutions: Optional[List[float]] = None,
    aggregation_method: str = 'mean',
    normalize: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dot_scale: Tuple[float, float] = (10, 200),
    colormap: str = 'viridis',
    show_resolution_labels: bool = True,
    cluster_spacing: float = 1.0,
    resolution_spacing: float = 3.0,
    show_cluster_colors: bool = True,
    cluster_color_width: float = 0.8,
    suptitle: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a dotplot showing cluster expression profiles across resolutions.
    
    Each dot represents a cluster at a specific resolution, where:
    - X-axis: Properties/genes
    - Y-axis: Clusters grouped by resolution
    - Dot size: Expression level (weighted mean)
    - Dot color: Expression level
    
    Parameters
    ----------
    expression_data : pd.DataFrame
        DataFrame with nodes/cells as rows and properties/genes as columns.
        Index should correspond to cell/node indices used in clustering.
        
    clustering_results : dict
        Dictionary mapping resolution values to cluster label arrays.
        Each array should have same length as expression_data rows.
        
    properties : list of str, optional
        List of column names from expression_data to include in plot.
        If None, uses all columns.
        
    resolutions : list of float, optional
        List of resolution values to include in plot.
        If None, uses all resolutions from clustering_results.
        
    aggregation_method : str, default='mean'
        Method to aggregate expression within clusters.
        Options: 'mean', 'median', 'sum'
        
    normalize : bool, default=True
        Whether to normalize expression values per property (0-1 scale).
        
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
        
    dot_scale : tuple, default=(10, 200)
        Range for dot sizes (min_size, max_size).
        
    colormap : str, default='viridis'
        Matplotlib colormap name.
        
    show_resolution_labels : bool, default=True
        Whether to show resolution values on y-axis.
        
    cluster_spacing : float, default=1.0
        Spacing between clusters within same resolution.
        
    resolution_spacing : float, default=3.0
        Additional spacing between different resolutions.
        
    show_cluster_colors : bool, default=True
        Whether to show a colored band on the left indicating cluster colors.
        Uses the same color scheme as multiresolution graph plots.
        
        cluster_color_width : float, default=0.8
        Height of the cluster color rectangles in the y-direction.    suptitle : str, optional
        Title for the entire figure.
        
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    **kwargs
        Additional arguments passed to scatter plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from treeclust.clustering import MultiresolutionLeiden
    >>> from treeclust.plotting import plot_cluster_expression_dotplot
    >>> 
    >>> # Create example expression data
    >>> expression_data = pd.DataFrame({
    ...     'gene1': np.random.randn(100),
    ...     'gene2': np.random.randn(100),
    ...     'gene3': np.random.randn(100)
    ... })
    >>> 
    >>> # Get clustering results
    >>> clustering = MultiresolutionLeiden()
    >>> clustering.fit(adjacency_matrix)
    >>> 
    >>> # Create dotplot
    >>> fig = plot_cluster_expression_dotplot(
    ...     expression_data=expression_data,
    ...     clustering_results=clustering.labels_,
    ...     properties=['gene1', 'gene2', 'gene3']
    ... )
    """
    # Input validation
    if not isinstance(expression_data, pd.DataFrame):
        raise ValueError("expression_data must be a pandas DataFrame")
    
    if not isinstance(clustering_results, dict):
        raise ValueError("clustering_results must be a dictionary")
    
    # Select properties to plot
    if properties is None:
        properties = list(expression_data.columns)
    else:
        missing_props = [p for p in properties if p not in expression_data.columns]
        if missing_props:
            raise ValueError(f"Properties not found in expression_data: {missing_props}")
    
    # Select resolutions to plot
    if resolutions is None:
        resolutions = sorted(clustering_results.keys())
    else:
        missing_res = [r for r in resolutions if r not in clustering_results]
        if missing_res:
            raise ValueError(f"Resolutions not found in clustering_results: {missing_res}")
    
    # Prepare aggregation function
    agg_functions = {
        'mean': 'mean',
        'median': 'median', 
        'sum': 'sum'
    }
    if aggregation_method not in agg_functions:
        raise ValueError(f"Unknown aggregation_method: {aggregation_method}. "
                        f"Options: {list(agg_functions.keys())}")
    agg_method_str = agg_functions[aggregation_method]
    
    # Extract expression data for selected properties
    expr_subset = expression_data[properties]
    
    # Compute cluster expression profiles
    cluster_profiles = []
    cluster_labels = []
    y_positions = []
    resolution_boundaries = []
    
    current_y = 0
    
    for res_idx, resolution in enumerate(resolutions):
        cluster_labels_res = clustering_results[resolution]
        
        # Check data alignment
        if len(cluster_labels_res) != len(expr_subset):
            raise ValueError(f"Cluster labels length ({len(cluster_labels_res)}) "
                           f"does not match expression data length ({len(expr_subset)}) "
                           f"for resolution {resolution}")
        
        # Get unique clusters for this resolution
        unique_clusters = sorted(np.unique(cluster_labels_res))
        
        # Store the position of the first cluster in this resolution
        first_cluster_y = current_y
        
        # Calculate and store resolution boundary if not the first resolution
        if res_idx > 0:
            # The boundary should be halfway between the last cluster of previous resolution
            # and the first cluster of current resolution
            last_cluster_prev = y_positions[-1] if y_positions else 0
            boundary_y = (last_cluster_prev + first_cluster_y) / 2
            resolution_boundaries.append(boundary_y)
        
        for cluster_idx, cluster_id in enumerate(unique_clusters):
            # Get cells in this cluster
            cluster_mask = cluster_labels_res == cluster_id
            cluster_cells = expr_subset.iloc[cluster_mask]
            
            if len(cluster_cells) == 0:
                warnings.warn(f"Empty cluster {cluster_id} at resolution {resolution}")
                continue
            
            # Aggregate expression for this cluster
            cluster_profile = cluster_cells.agg(agg_method_str, axis=0)
            cluster_profiles.append(cluster_profile.values)
            
            # Create label and position
            cluster_label = f"R{resolution}_C{cluster_id}"
            cluster_labels.append(cluster_label)
            y_positions.append(current_y)
            
            current_y += cluster_spacing
        
        # Add extra spacing between resolutions
        if res_idx < len(resolutions) - 1:
            current_y += resolution_spacing - cluster_spacing
    
    if not cluster_profiles:
        raise ValueError("No valid clusters found to plot")
    
    # Convert to arrays
    cluster_profiles = np.array(cluster_profiles)  # Shape: (n_clusters, n_properties)
    y_positions = np.array(y_positions)
    
    # Create cluster color mapping if requested
    cluster_colors = []
    if show_cluster_colors:
        # Collect all unique cluster IDs across all resolutions
        all_clusters = set()
        for resolution in resolutions:
            cluster_labels_res = clustering_results[resolution]
            all_clusters.update(np.unique(cluster_labels_res))
        all_clusters = sorted(all_clusters)
        
        # Create color palette consistent with multiresolution graph plots
        if all_clusters and all(isinstance(cluster, (int, np.integer)) for cluster in all_clusters):
            n_clusters = len(all_clusters)
            if n_clusters <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
            elif n_clusters <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
            else:
                colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
            
            global_color_map = dict(zip(all_clusters, colors))
            
            # Assign colors to each cluster in the plot
            for label in cluster_labels:
                # Extract cluster ID from label (format: "R{resolution}_C{cluster_id}")
                cluster_id = int(label.split('_')[1][1:])  # Remove 'C' prefix and convert to int
                cluster_colors.append(global_color_map[cluster_id])
        else:
            # Fallback to default colors if cluster IDs are not integers
            cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_labels)))
    
    # Normalize expression values if requested
    if normalize:
        # Normalize each property (column) to 0-1 scale
        cluster_profiles = (cluster_profiles - cluster_profiles.min(axis=0)) / \
                          (cluster_profiles.max(axis=0) - cluster_profiles.min(axis=0) + 1e-8)
    
    # Create figure if needed
    if ax is None:
        if figsize is None:
            # Auto-calculate figure size
            width = max(8, len(properties) * 0.8)
            height = max(6, len(cluster_labels) * 0.3)
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create coordinate grids for plotting
    x_coords = []
    y_coords = []
    sizes = []
    colors = []
    
    for prop_idx, property_name in enumerate(properties):
        for cluster_idx in range(len(cluster_profiles)):
            expr_value = cluster_profiles[cluster_idx, prop_idx]
            
            x_coords.append(prop_idx)
            y_coords.append(y_positions[cluster_idx])
            
            # Size and color based on expression level
            if normalize:
                # Normalized values are already 0-1
                size_factor = expr_value
                color_value = expr_value
            else:
                # Use min-max normalization for sizing/coloring
                prop_values = cluster_profiles[:, prop_idx]
                size_factor = (expr_value - prop_values.min()) / (prop_values.max() - prop_values.min() + 1e-8)
                color_value = size_factor
            
            # Map to dot size range
            dot_size = dot_scale[0] + (dot_scale[1] - dot_scale[0]) * size_factor
            sizes.append(dot_size)
            colors.append(color_value)
    
    # Add cluster color band if requested
    if show_cluster_colors and cluster_colors:
        # Create rectangles for each cluster showing its color
        band_width = 0.3  # Horizontal width (shorter in x-axis)
        for i, (y_pos, color) in enumerate(zip(y_positions, cluster_colors)):
            rect = plt.Rectangle(
                (-1.1, y_pos - cluster_color_width/2), 
                band_width,  # Shorter width in x-direction
                cluster_color_width,  # Height in y-direction 
                facecolor=color, 
                edgecolor='black', 
                linewidth=0.3,
                alpha=0.9
            )
            ax.add_patch(rect)
    
    # Create scatter plot (remove figure-level parameters from kwargs)
    scatter_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['suptitle']}  # Remove any figure-level params
    
    scatter = ax.scatter(x_coords, y_coords, s=sizes, c=colors, 
                        cmap=colormap, alpha=0.7, edgecolors='black', 
                        linewidth=0.5, **scatter_kwargs)
    
    # Customize axes
    ax.set_xticks(range(len(properties)))
    ax.set_xticklabels(properties, rotation=45, ha='right')
    ax.set_xlabel('Properties/Genes')
    
    # Adjust x-axis limits to accommodate color band and resolution labels
    if show_cluster_colors:
        ax.set_xlim(-1.4, len(properties) - 0.5)  # Space for color band
    elif show_resolution_labels:
        ax.set_xlim(-0.8, len(properties) - 0.5)  # Space for resolution labels only
    else:
        ax.set_xlim(-0.5, len(properties) - 0.5)
    
    # Y-axis labels - show only cluster IDs on ticks
    ax.set_yticks(y_positions)
    if show_resolution_labels:
        # Show only cluster IDs (C0, C1, etc.) on the tick labels
        y_labels = []
        resolution_positions = {}  # Track positions for each resolution
        current_resolution = None
        resolution_y_positions = []
        
        for i, label in enumerate(cluster_labels):
            parts = label.split('_')
            resolution_str = parts[0]  # R0, R1, R2, etc.
            cluster_str = parts[1]     # C0
            
            # Extract resolution index (now an integer)
            resolution_idx = int(resolution_str[1:])  # Remove 'R' prefix and convert to int
            
            # Just show cluster ID
            y_labels.append(cluster_str)
            
            # Track resolution positions for middle positioning
            if resolution_idx != current_resolution:
                if current_resolution is not None:
                    # Store the middle position of the previous resolution
                    resolution_positions[current_resolution] = np.mean(resolution_y_positions)
                current_resolution = resolution_idx
                resolution_y_positions = []
            
            resolution_y_positions.append(y_positions[i])
        
        # Don't forget the last resolution
        if current_resolution is not None:
            resolution_positions[current_resolution] = np.mean(resolution_y_positions)
        
        ax.set_yticklabels(y_labels, fontsize=8, va='center')
        
        # Add vertical resolution labels positioned next to y-tick labels (outside axes)
        for resolution_idx, middle_y in resolution_positions.items():
            ax.text(-0.05, middle_y, f'R{resolution_idx}', 
                   rotation=90, ha='right', va='center', 
                   fontsize=10, color='black',
                   transform=ax.get_yaxis_transform())
    else:
        # Show only cluster IDs without resolution info
        y_labels = [label.split('_')[1] for label in cluster_labels]
        ax.set_yticklabels(y_labels, va='center')
    
    # Add "Clusters" as an x-tick label on the left
    current_xticks = list(ax.get_xticks())
    current_xlabels = list(ax.get_xticklabels())
    
    # Add "Clusters" at position -1
    new_xticks = [-1] + current_xticks
    new_xlabels = ['Clusters'] + [label.get_text() for label in current_xlabels]
    
    ax.set_xticks(new_xticks)
    ax.set_xticklabels(new_xlabels, rotation=45, ha='right')
    
    # Move ylabel further to avoid collision with resolution labels  
    ax.set_ylabel('Clusters by Resolution', labelpad=30)
    
    # Add resolution boundaries with more prominent lines
    for boundary in resolution_boundaries:
        ax.axhline(y=boundary, color='darkgray', linestyle='-', alpha=0.8, linewidth=2)
        # Add a subtle background highlight for resolution separation
        if show_cluster_colors:
            ax.axhline(y=boundary, color='darkgray', linestyle='-', alpha=0.3, linewidth=6)
    
    # Add resolution boundaries with vertical lines instead of horizontal
    for boundary in resolution_boundaries:
        ax.axvline(x=boundary, color='darkgray', linestyle='-', alpha=0.8, linewidth=2)
        # Add a subtle background highlight for resolution separation
        if show_cluster_colors:
            ax.axvline(x=boundary, color='darkgray', linestyle='-', alpha=0.3, linewidth=6)
    
    # Add legend for color scale and size
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    # Create legend elements for color scale
    color_min, color_max = np.min(colors), np.max(colors)
    color_range = np.linspace(color_min, color_max, 5)
    color_elements = []
    for val in color_range:
        color = plt.cm.viridis(plt.Normalize(color_min, color_max)(val))
        color_elements.append(Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color, markersize=8, 
                                   label=f'{val:.2f}'))
    
    # Create legend elements for size scale if sizes vary
    size_elements = []
    if len(np.unique(sizes)) > 1:
        size_min, size_max = np.min(sizes), np.max(sizes)
        size_range = np.linspace(size_min, size_max, 3)
        for val in size_range:
            # Scale marker size for legend (make it visible but proportional)
            legend_size = 4 + (val - size_min) / (size_max - size_min) * 8
            size_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor='gray', markersize=legend_size, 
                                      label=f'{val:.1f}'))
    
    # Combine all legend elements
    all_elements = []
    if color_elements:
        all_elements.extend(color_elements)
        all_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
    
    if size_elements:
        all_elements.append(Line2D([0], [0], color='none', 
                                 label=f'Size ({size_elements[0].get_label().split()[0]} Scale)'))
        all_elements.extend(size_elements)
    
    # Add single legend outside the plot
    if all_elements:
        legend = ax.legend(handles=all_elements, 
                          title='Expression & Size Scale',
                          loc='center left', 
                          bbox_to_anchor=(1.05, 0.5),
                          frameon=True,
                          fancybox=True,
                          shadow=True)
    
    # Invert y-axis to show higher resolutions at top
    ax.invert_yaxis()
    
    # Add suptitle if provided
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14, y=0.98)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    if 'all_elements' in locals() and all_elements:
        plt.subplots_adjust(right=0.75)  # Make room for legend
    
    return fig
