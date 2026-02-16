import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from typing import Optional, Tuple, Union, Dict, List
import copy

from .graph import plot_graph


def _add_cluster_labels(ax, node_pos, cluster_labels, min_distance=0.1):
    """
    Add cluster labels to a plot, positioned at cluster centroids with collision avoidance.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add labels to.
    node_pos : np.ndarray
        Node positions as (n_nodes, 2) array.
    cluster_labels : array-like
        Cluster labels for each node.
    min_distance : float, default=0.1
        Minimum distance between labels to avoid overlapping.
    """
    if node_pos is None or cluster_labels is None:
        return
        
    # Convert to numpy arrays for easier computation
    positions = np.array(node_pos)
    labels = np.array(cluster_labels)
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    
    # Calculate cluster centroids
    cluster_centroids = {}
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        if np.any(mask):
            centroid = np.mean(positions[mask], axis=0)
            cluster_centroids[cluster_id] = centroid
    
    # Adjust label positions to avoid overlaps
    adjusted_positions = _adjust_label_positions(
        list(cluster_centroids.values()), 
        min_distance
    )
    
    # Add text labels
    for i, cluster_id in enumerate(cluster_centroids.keys()):
        pos = adjusted_positions[i]
        ax.text(
            pos[0], pos[1], 
            str(cluster_id),
            fontsize=12,
            fontweight='bold',
            ha='center', va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='black',
                alpha=0.8
            ),
            zorder=1000  # Ensure labels appear on top
        )


def _adjust_label_positions(positions, min_distance=0.1, max_iterations=100):
    """
    Adjust label positions to avoid overlapping using a simple repulsion algorithm.
    
    Parameters
    ----------
    positions : list of array-like
        Original positions as [(x, y), ...].
    min_distance : float
        Minimum distance between labels.
    max_iterations : int
        Maximum number of iterations for adjustment.
        
    Returns
    -------
    adjusted_positions : list of np.ndarray
        Adjusted positions.
    """
    if len(positions) <= 1:
        return [np.array(pos) for pos in positions]
    
    # Convert to numpy array for easier computation
    pos_array = np.array(positions)
    adjusted = pos_array.copy()
    
    for iteration in range(max_iterations):
        moved = False
        
        for i in range(len(adjusted)):
            for j in range(i + 1, len(adjusted)):
                # Calculate distance between labels i and j
                diff = adjusted[i] - adjusted[j]
                distance = np.linalg.norm(diff)
                
                if distance < min_distance and distance > 0:
                    # Calculate repulsion vector
                    unit_vector = diff / distance
                    overlap = min_distance - distance
                    
                    # Move both labels away from each other
                    move_vector = unit_vector * overlap * 0.5
                    adjusted[i] += move_vector
                    adjusted[j] -= move_vector
                    moved = True
        
        if not moved:
            break
    
    return [pos for pos in adjusted]


def plot_multiresolution_graph(
    node_pos: Optional[Union[np.ndarray, Dict]],
    edge_width: Union[np.ndarray, sparse.spmatrix, Dict[float, np.ndarray]] = None,
    node_color: Union[Dict[float, np.ndarray], np.ndarray] = None,
    node_size: Union[int, float, Dict[float, np.ndarray], np.ndarray] = 50,
    node_size_norm: Optional[Tuple[float, float]] = None,
    node_sizes: Optional[Tuple[float, float]] = None,
    edge_width_norm: Optional[Tuple[float, float]] = None,
    edge_widths: Optional[Tuple[float, float]] = None,
    resolutions_to_plot: Optional[List[float]] = None,
    max_plots: int = 9,
    figsize: Optional[Tuple[float, float]] = None,
    layout: str = 'spring',
    edge_alpha: float = 0.3,
    show_cluster_labels: bool = True,
    suptitle: Optional[str] = None
) -> plt.Figure:
    """
    Plot clustering results across multiple resolutions with seaborn-style data specification.
    
    This function creates a grid of subplots showing clustering results at different
    resolution values following seaborn's scatterplot API pattern for intuitive usage.
    
    Parameters
    ----------
    edge_width : np.ndarray, scipy.sparse matrix, or dict
        Edge connectivity and/or weight data. Can be:
        - Matrix: Edge weights/connectivity for plotting edges  
        - Dict mapping resolution -> edge matrix: Per-resolution edge data
        - If None, will use adjacency_matrix (deprecated)
        
    node_color : dict or np.ndarray
        Node coloring data (primary parameter, replaces labels). Can be:
        - Dict mapping resolution -> color data: Multi-resolution colors/labels
        - Single array: Color data for single resolution plotting
        - If None, will use labels parameter (deprecated)
        
    node_pos : np.ndarray or dict, optional
        Node positions as (n_nodes, 2) array or dict. If None, will be computed
        using the specified layout algorithm.
        
    node_size : int, float, dict, or np.ndarray, default=50
        Node sizing data. Can be:
        - Single number: Fixed size for all nodes
        - Dict mapping resolution -> size data: Per-resolution sizing (e.g., confidence scores)
        - Single array: Size data for single resolution plotting
        
    node_size_norm : tuple of float, optional
        Normalization range for node_size data, e.g., (0, 1) for confidence scores.
        If None, will use data min/max range.
        
    node_sizes : tuple of float, optional
        Size range (min_size, max_size) for mapping normalized node_size to actual sizes.
        Default is (20, 100). Like seaborn's sizes parameter.
        
    adjacency_matrix : np.ndarray or scipy.sparse matrix, optional
        DEPRECATED: Use edge_width instead. Adjacency matrix for backward compatibility.
        
    labels : dict or np.ndarray, optional  
        DEPRECATED: Use node_color instead. Cluster labels for backward compatibility.
        
    edge_width_norm : tuple of float, optional
        Normalization range for edge_width data.
        If None, will use data min/max range.
        
    edge_widths : tuple of float, optional
        Width range (min_width, max_width) for mapping normalized edge_width to actual widths.
        Default is (0.1, 2.0).
        
    resolutions_to_plot : list of float, optional
        Specific resolution values to plot. If None, will plot all available
        resolutions up to max_plots.
        
    max_plots : int, default=9
        Maximum number of subplots to create.
        
    figsize : tuple of float, optional
        Figure size (width, height) in inches. If None, will be computed
        automatically based on number of subplots.
        
    layout : str, default='spring'
        Layout algorithm for node positioning if node_pos is None.
        
    edge_alpha : float, default=0.3
        Transparency of edges.
        
    show_cluster_labels : bool, default=True
        Whether to display cluster ID labels on top of cluster positions.
        Labels are positioned at cluster centroids and automatically repositioned
        to avoid overlapping when clusters are too close together.
        
    suptitle : str, optional
        Super title for the entire figure. If None, will use default title.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plots.
        
    Examples
    --------
    # Multiresolution with confidence-based sizing (new seaborn-style API)
    >>> fig = plot_multiresolution_clustering(
    ...     edge_width=matrix,                   # Edge connectivity/weights
    ...     node_color=results.labels_,          # Cluster labels for coloring
    ...     node_size=results.cell_confidence_,  # Confidence data values
    ...     node_size_norm=(0, 1),               # Normalization range
    ...     node_sizes=(10, 50),                 # Size mapping
    ...     node_pos=X[:, :2]
    ... )
    
    # With edge weights and custom ranges
    >>> fig = plot_multiresolution_clustering(
    ...     edge_width=weighted_matrix,          # Edge weights
    ...     node_color=results.labels_,          # Cluster labels
    ...     node_size=results.cell_confidence_,
    ...     node_size_norm=(0, 1),              # Confidence range
    ...     node_sizes=(20, 100),               # Node size range
    ...     edge_width_norm=(0, 1),             # Edge weight range
    ...     edge_widths=(0.5, 3.0)              # Edge width range
    ... )
    
    # Single resolution with custom normalization
    >>> fig = plot_multiresolution_clustering(
    ...     edge_width=matrix,                   # Single matrix
    ...     node_color=cluster_labels,           # Single array
    ...     node_size=stability_scores,          # Single array
    ...     node_size_norm=(0.5, 1.0),         # Custom normalization
    ...     node_sizes=(15, 75)                 # Custom size range
    ... )
    """
    # Handle backward compatibility and parameter precedence
    # Primary: edge_width, fallback: adjacency_matrix
    if edge_width is not None:
        connectivity_matrix = edge_width
    else:
        raise ValueError("edge_width must be provided")
    
    # Handle single vs multiresolution case for determining which resolutions to plot
    if node_color is not None:
        if isinstance(node_color, dict):
            # Multiresolution case
            available_resolutions = [res for res, data in node_color.items() if data is not None]
        else:
            # Single resolution case - create dummy resolution
            available_resolutions = [1.0]
    else:
        # No color data provided, assume single resolution
        available_resolutions = [1.0]
    
    if not available_resolutions:
        raise ValueError("No valid clustering results found to plot.")
    
    # Filter to specific resolutions if requested
    if resolutions_to_plot is not None:
        available_resolutions = [res for res in resolutions_to_plot if res in available_resolutions]
    
    # Limit number of plots
    if len(available_resolutions) > max_plots:
        available_resolutions = available_resolutions[:max_plots]
    
    # Compute grid size
    n_plots = len(available_resolutions)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
        
    # Handle node colors - simple pass-through to plot_graph
    if node_color is not None:
        if isinstance(node_color, dict):
            colors_dict = node_color
        elif len(available_resolutions) == 1:
            # Single array for single resolution
            colors_dict = {available_resolutions[0]: node_color}
        else:
            colors_dict = {}
    else:
        colors_dict = {}
    
    # Create global color mapping for consistent colors based on cluster ID order
    global_color_map = None
    if colors_dict:
        # Collect all unique cluster labels across resolutions
        all_clusters = set()
        for res_colors in colors_dict.values():
            if res_colors is not None:
                # Check if these are cluster labels (integers)
                if hasattr(res_colors, '__iter__') and not isinstance(res_colors, str):
                    sample_values = list(res_colors)[:10]  # Check first 10 values
                    if all(isinstance(val, (int, np.integer)) for val in sample_values):
                        all_clusters.update(res_colors)
        
        # Create consistent color mapping based on cluster ID order
        if all_clusters:
            # Sort clusters by ID (0, 1, 2, 3, 4...)
            sorted_clusters = sorted(all_clusters)
            n_clusters = len(sorted_clusters)
            
            # Choose appropriate colormap
            if n_clusters <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
            elif n_clusters <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]  
            else:
                colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
            
            # Map cluster IDs to colors in order: 0->color0, 1->color1, 2->color2, etc.
            global_color_map = dict(zip(sorted_clusters, colors))
    
    # Handle node sizes
    if isinstance(node_size, dict):
        sizes_dict = node_size
    elif isinstance(node_size, (list, np.ndarray)) and len(available_resolutions) == 1:
        # Single array for single resolution
        sizes_dict = {available_resolutions[0]: node_size}
    else:
        # Fixed size for all
        sizes_dict = {}
    
    # Plot each resolution
    for i, resolution in enumerate(available_resolutions):
        ax = axes[i]
        
        # Get colors for this resolution
        if resolution in colors_dict:
            node_colors = colors_dict[resolution]
            
            # Apply global color mapping if available
            if global_color_map is not None and node_colors is not None:
                # Convert cluster labels to actual colors using global mapping
                if hasattr(node_colors, '__iter__') and not isinstance(node_colors, str):
                    node_colors = [global_color_map.get(label, 'gray') for label in node_colors]
        else:
            node_colors = None  # Let plot_graph handle defaults
        
        # Get sizes for this resolution - pass raw data to plot_graph
        if resolution in sizes_dict:
            node_size_data = sizes_dict[resolution]
        else:
            node_size_data = node_size
        
        # Create the plot - let plot_graph handle all normalization
        ax_out = plot_graph(
            connectivity_matrix,
            node_color=node_colors, 
            node_pos=node_pos, 
            node_size=node_size_data,
            node_size_norm=node_size_norm,
            node_sizes=node_sizes,
            edge_width_norm=edge_width_norm,
            edge_widths=edge_widths,
            edge_alpha=edge_alpha,
            ax=ax,
            legend=False  # Handle legend centrally
        )
        
        # Add cluster labels if requested (use original labels for labeling)
        if show_cluster_labels and resolution in colors_dict and colors_dict[resolution] is not None:
            _add_cluster_labels(ax, node_pos, colors_dict[resolution])
        
        # Set title (hide resolution for single resolution case)
        if len(available_resolutions) == 1:
            ax.set_title("Clustering Results")
        else:
            ax.set_title(f"Resolution {resolution}")
        ax.axis('off')
    
    # Remove unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig
