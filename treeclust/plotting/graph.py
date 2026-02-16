import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy import sparse
from typing import Optional, Tuple, Union, Dict, List, Any
import warnings

from distinctipy import distinctipy

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not available. Some layout algorithms will not work.")


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


def plot_graph(
    node_pos: Optional[Union[np.ndarray, Dict]],
    edge_width: Union[np.ndarray, sparse.spmatrix] = None,
    node_color: Optional[Union[str, np.ndarray, List]] = None,
    node_size: Union[int, float, np.ndarray] = 50,
    node_size_norm: Optional[Tuple[float, float]] = None,
    node_sizes: Optional[Tuple[float, float]] = None,
    node_cmap: Optional[Any] = None,
    edge_width_norm: Optional[Tuple[float, float]] = None,
    edge_widths: Optional[Tuple[float, float]] = None,
    edge_alpha: float = 0.5,
    layout: str = 'spring',
    layout_kwargs: Optional[Dict] = None,
    figsize: Tuple[float, float] = (10, 8),
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Plot a graph using seaborn-style data specification.
    
    This function provides a seaborn-like interface for graph visualization with
    separate parameters for data, normalization, and visual mapping.
    
    Parameters
    ----------
    edge_width : np.ndarray or scipy.sparse matrix
        Edge connectivity and/or weight data. Matrix of shape (n_nodes, n_nodes).
        Can be binary (connectivity) or weighted.
        
    node_pos : np.ndarray or dict, optional
        Node positions as (n_nodes, 2) array or dict mapping node indices to (x, y).
        If None, positions will be computed using the specified layout algorithm.
        
    node_color : str, np.ndarray, or list, optional
        Node coloring data. Can be:
        - String: Single color for all nodes
        - Array: Color/label data for each node
        - List: Explicit colors for each node
        
    node_size : int, float, or np.ndarray, default=50
        Node sizing data. Can be:
        - Single number: Fixed size for all nodes
        - Array: Size data for each node (will be normalized and scaled)
        
    node_size_norm : tuple of float, optional
        Normalization range for node_size data, e.g., (0, 1) for confidence scores.
        If None, will use data min/max range.
        
    node_sizes : tuple of float, optional
        Size range (min_size, max_size) for mapping normalized node_size to actual sizes.
        Default is (20, 100). Like seaborn's sizes parameter.
        
    edge_width_norm : tuple of float, optional
        Normalization range for edge_width data.
        If None, will use data min/max range for edge weights.
        
    edge_widths : tuple of float, optional
        Width range (min_width, max_width) for mapping normalized edge weights to line widths.
        Default is (0.1, 2.0).
        
    edge_alpha : float, default=0.5
        Transparency of edges (0=transparent, 1=opaque).
        
    layout : str, default='spring'
        Layout algorithm for positioning nodes. Options:
        - 'spring': Spring-force layout (requires NetworkX)
        - 'circular': Circular layout
        - 'random': Random positions
        - 'grid': Grid layout (for small graphs)
        
    layout_kwargs : dict, optional
        Additional keyword arguments for layout algorithm.
        
    figsize : tuple, default=(10, 8)
        Figure size as (width, height).
        
    legend : bool, default=True
        Whether to show a legend/colorbar for node colors. For continuous data,
        shows a colorbar. For discrete categories, shows a categorical legend.
        
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
        
    show_labels : bool, default=False
        Whether to show text labels on nodes. Labels will display the node_color values
        (e.g., cluster IDs, prediction labels, etc.) as text on each node.
                
    **kwargs : dict
        Additional keyword arguments passed to scatter plot for nodes.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
        
    Examples
    --------
    # Basic graph with edge weights (new seaborn-style API)
    >>> fig = plot_graph(
    ...     edge_width=adjacency_matrix,     # Edge connectivity/weights
    ...     node_color=cluster_labels,       # Node colors from labels
    ...     node_size=confidence_scores,     # Node sizes from confidence
    ...     node_size_norm=(0, 1),          # Confidence normalization
    ...     node_sizes=(20, 100),           # Size mapping
    ...     node_pos=positions
    ... )
    
    # With custom edge width scaling
    >>> fig = plot_graph(
    ...     edge_width=weighted_matrix,      # Edge weights
    ...     node_color=labels,
    ...     edge_width_norm=(0, 1),         # Edge weight normalization
    ...     edge_widths=(0.5, 3.0),         # Edge width mapping
    ...     node_size=50                    # Fixed node size
    ... )
    """
    # Handle backward compatibility and parameter precedence
    # Primary: edge_width, fallback: adjacency_matrix
    if edge_width is not None:
        connectivity_matrix = edge_width
    else:
        raise ValueError("edge_width must be provided")
    
    # Handle edge width scaling from connectivity matrix
    processed_edge_width = 1.0  # Default edge width
    if edge_width_norm is not None or edge_widths is not None:
        # Extract edge weights from connectivity matrix for scaling
        if sparse.issparse(connectivity_matrix):
            # Get non-zero edge weights
            coo = connectivity_matrix.tocoo()
            edge_weights = coo.data[coo.data > 0]
        else:
            # Get non-zero edge weights from dense matrix
            edge_weights = connectivity_matrix[connectivity_matrix > 0]
        
        if len(edge_weights) > 0:
            # Set default width range if not provided
            if edge_widths is None:
                width_range = (0.1, 2.0)
            else:
                width_range = edge_widths
            
            # Set default normalization range if not provided
            if edge_width_norm is None:
                norm_range = (edge_weights.min(), edge_weights.max())
            else:
                norm_range = edge_width_norm
            
            # Normalize edge weights
            norm_min, norm_max = norm_range
            if norm_min == norm_max:
                normalized_weights = np.ones_like(edge_weights) * 0.5
            else:
                normalized_weights = (edge_weights - norm_min) / (norm_max - norm_min)
                normalized_weights = np.clip(normalized_weights, 0, 1)
            
            # Map to width range
            min_width, max_width = width_range
            processed_edge_width = min_width + (max_width - min_width) * normalized_weights
    
    # Convert sparse matrix to dense if needed for easier processing
    if sparse.issparse(connectivity_matrix):
        adj_dense = connectivity_matrix.toarray()
        edges = connectivity_matrix
    else:
        adj_dense = connectivity_matrix
        edges = sparse.csr_matrix(connectivity_matrix)
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Handle discrete colormaps for integer colors and string labels
    if node_cmap is None and node_color is not None:
        if len(np.unique(node_color)) <= 10:
            node_cmap = "tab10"
        elif len(np.unique(node_color)) <= 20:
            node_cmap = "tab20"
        else:
            node_cmap = distinctipy.get_colors(len(np.unique(node_color)), pastel_factor=0.3, rng=0)

    original_labels = None  # Store original labels for text display
    
    # Plot edges (lower z-order so they appear behind nodes)
    _plot_edges(ax, edges, node_pos, edge_alpha, processed_edge_width)

    # Plot nodes (higher z-order so they appear on top of edges)
    sns.scatterplot(
        x=node_pos[:, 0], y=node_pos[:, 1],
        size=node_sizes,
        size_norm=node_size_norm,
        sizes=node_sizes,
        hue=node_color,
        hue_order=np.sort(np.unique(node_color)) if node_color is not None else None,
        palette=node_cmap,
        zorder=2,  # Higher z-order to appear on top
        ax=ax,
        **kwargs
    )
    
    # Add text labels on nodes if requested
    if show_labels and node_color is not None:
        # Use original string labels if available, otherwise use current node_color values
        labels_to_show = original_labels if original_labels is not None else node_color
        
        if hasattr(labels_to_show, '__iter__') and not isinstance(labels_to_show, str):
            # Use the same cluster labeling approach as multiresolution plots
            _add_cluster_labels(ax, node_pos, labels_to_show)
        
    return fig

def _plot_edges(
    ax: plt.Axes,
    edges: sparse.spmatrix,
    pos: np.ndarray,
    alpha: float,
    width: Union[float, np.ndarray]
) -> None:
    """Plot edges as line segments using optimized vectorized operations."""
    # Convert to COO format for efficient access to indices and data
    edges_coo = edges.tocoo()
    
    if edges_coo.nnz == 0:
        return  # No edges to plot
    
    # Get source and target indices
    sources = edges_coo.row
    targets = edges_coo.col
    
    # For undirected graphs, only plot each edge once (source < target)
    mask = sources < targets
    if np.any(mask):
        sources = sources[mask]
        targets = targets[mask]
        edge_data = edges_coo.data[mask] if hasattr(edges_coo, 'data') else None
    else:
        # All edges are upper triangular already or directed graph
        edge_data = edges_coo.data if hasattr(edges_coo, 'data') else None
    
    if len(sources) == 0:
        return
    
    # Vectorized creation of line segments
    # Shape: (n_edges, 2, 2) where each edge has start and end points with x,y coords
    segments = np.stack([pos[sources], pos[targets]], axis=1)
    
    # Handle edge widths
    if np.isscalar(width):
        linewidths = width
    else:
        linewidths = np.array(width)
        # Only check length if linewidths is an array
        if hasattr(linewidths, '__len__') and len(linewidths) != len(segments):
            linewidths = linewidths[:len(segments)]
    
    # Create line collection with lower z-order to appear behind nodes
    lc = LineCollection(segments, alpha=alpha, linewidths=linewidths, colors='gray', zorder=1)
    ax.add_collection(lc)

def plot_clustering_comparison(
    adjacency_matrix: Union[np.ndarray, sparse.spmatrix],
    clustering_results: Dict[str, np.ndarray],
    pos: Optional[np.ndarray] = None,
    layout: str = 'spring',
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple clustering results side by side for comparison.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray or scipy.sparse matrix
        Adjacency matrix for the graph structure.
        
    clustering_results : dict
        Dictionary mapping method names to cluster label arrays.
        
    pos : np.ndarray, optional
        Node positions. If None, computed using layout algorithm.
        
    layout : str, default='spring'
        Layout algorithm for node positioning.
        
    figsize : tuple, optional
        Figure size. If None, computed based on number of methods.
        
    suptitle : str, optional
        Overall title for the comparison plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing all comparison plots.
        
    Examples
    --------
    >>> from treeclust.plotting import plot_clustering_comparison
    >>> from treeclust.clustering import Leiden, Louvain
    >>> 
    >>> # Multiple clustering results
    >>> leiden = Leiden(resolution=1.0)
    >>> louvain = Louvain(resolution=1.0)
    >>> 
    >>> results = {
    ...     'Leiden': leiden.fit_predict(adjacency),
    ...     'Louvain': louvain.fit_predict(adjacency)
    ... }
    >>> 
    >>> # Compare results
    >>> fig = plot_clustering_comparison(adjacency, results)
    >>> plt.show()
    """
    n_methods = len(clustering_results)
    
    if figsize is None:
        figsize = (5 * n_methods, 5)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
        
    for i, (method_name, labels) in enumerate(clustering_results.items()):
        ax = axes[i]
        
        # Plot on specific axes
        plot_graph(
            adjacency_matrix,
            pos=pos,
            labels=labels,
            title=f'{method_name}\n({len(np.unique(labels))} clusters)',
            ax=ax,
            legend=False  # Disable individual legends
        )
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_multiresolution_analysis(
    multiresolution_result,
    adjacency_matrix: Union[np.ndarray, sparse.spmatrix],
    pos: Optional[np.ndarray] = None,
    resolutions_to_plot: Optional[List[float]] = None,
    max_plots: int = 6,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot clustering results across multiple resolution values.
    
    Parameters
    ----------
    multiresolution_result : MultiresolutionLeiden or MultiresolutionLouvain
        Fitted multiresolution clustering object.
        
    adjacency_matrix : np.ndarray or scipy.sparse matrix
        Adjacency matrix used for clustering.
        
    pos : np.ndarray, optional
        Node positions. If None, computed from data.
        
    resolutions_to_plot : list, optional
        Specific resolution values to plot. If None, selects representative values.
        
    max_plots : int, default=6
        Maximum number of resolution plots to show.
        
    figsize : tuple, optional
        Figure size. If None, computed based on number of plots.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing multiresolution analysis.
        
    Examples
    --------
    >>> from treeclust.clustering import MultiresolutionLeiden
    >>> from treeclust.plotting import plot_multiresolution_analysis
    >>> 
    >>> # Multiresolution clustering
    >>> mr_leiden = MultiresolutionLeiden(
    ...     resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0]
    ... )
    >>> mr_leiden.fit(adjacency)
    >>> 
    >>> # Plot results
    >>> fig = plot_multiresolution_analysis(mr_leiden, adjacency)
    >>> plt.show()
    """
    if not hasattr(multiresolution_result, 'labels_') or not multiresolution_result.is_fitted_:
        raise ValueError("Multiresolution object must be fitted before plotting")
    
    # Select resolutions to plot
    if resolutions_to_plot is None:
        available_resolutions = [
            res for res, labels in multiresolution_result.labels_.items() 
            if labels is not None
        ]
        
        if len(available_resolutions) <= max_plots:
            resolutions_to_plot = sorted(available_resolutions)
        else:
            # Select representative resolutions
            indices = np.linspace(0, len(available_resolutions) - 1, max_plots, dtype=int)
            resolutions_to_plot = [available_resolutions[i] for i in indices]
    
    n_plots = len(resolutions_to_plot)
    
    if figsize is None:
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        figsize = (4 * cols, 4 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
        
    for i, resolution in enumerate(resolutions_to_plot):
        if i >= len(axes):
            break
            
        labels = multiresolution_result.get_labels(resolution)
        
        if labels is not None:
            n_clusters = len(np.unique(labels))
            title = f'Resolution {resolution:.1f}\n({n_clusters} clusters)'
            
            plot_graph(
                adjacency_matrix,
                pos=pos,
                labels=labels,
                title=title,
                ax=axes[i],
                legend=False,
                node_size=30
            )
    
    # Hide empty subplots
    for i in range(len(resolutions_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Multiresolution Clustering Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

