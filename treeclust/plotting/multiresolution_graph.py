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
    warnings.warn("NetworkX not available. Some layout layout algorithms will not work.")

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    warnings.warn("igraph not available. Tree layout algorithm will use fallback implementation.")

def plot_multiresolution_hierarchy(
    labels_dict: Dict[Any, np.ndarray],
    color_dict: Optional[Dict[Tuple[Any, Any], Union[str, int, float]]] = None,
    size_dict: Optional[Dict[Tuple[Any, Any], Union[int, float]]] = None,
    min_overlap: float = 0.0,
    figsize: Tuple[float, float] = (12, 8),
    node_size_range: Tuple[float, float] = (50, 500),
    ax: Optional[plt.Axes] = None,
    layout_algorithm: str = 'igraph_tree',
    show_labels: bool = True,
    edge_alpha: float = 0.6,
    edge_style: str = 'bezier',
    colorbar: bool = True,
    cmap: Any = None,
    **kwargs
) -> plt.Figure:
    """
    Plot hierarchical tree of clusters across resolutions for multiresolution clustering.
    
    Shows how clusters split and merge across different resolution values, with
    connections based on shared cell membership. Node color and size can be specified
    directly through dictionaries.
    
    Parameters
    ----------
    labels_dict : dict
        Dictionary mapping resolution indices to cluster label arrays.
        Example: {0: array([0, 0, 1, 1, 2]), 1: array([0, 1, 1, 2, 2])}
        
    color_dict : dict, optional
        Dictionary mapping (resolution_idx, cluster_id) to color values.
        Values can be strings (color names), integers (categorical), or floats (continuous).
        Example: {(0, 0): 'red', (0, 1): 'blue', (1, 0): 0.5, (1, 1): 0.8}
        If None, colors by cluster labels (discrete coloring).
        
    size_dict : dict, optional
        Dictionary mapping (resolution_idx, cluster_id) to size values (numeric).
        Example: {(0, 0): 100, (0, 1): 50, (1, 0): 75, (1, 1): 200}
        If None, sizes by cluster size.
        
    min_overlap : float, default=0.0
        Minimum Jaccard similarity threshold for connecting clusters between resolutions.
        
    figsize : tuple, default=(12, 8)
        Figure size in inches.
        
    node_size_range : tuple, default=(50, 500)
        Range for node sizes (min_size, max_size).
        
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    layout_algorithm : str, default='igraph_tree'
        Layout algorithm for positioning nodes:
        - 'igraph_tree': Use igraph's optimized tree layout algorithms (requires igraph)
        - 'organic': Natural tree layout with splits emanating from parent centers
        - 'tree': Hierarchical tree layout (minimizes crossings)
        - 'spring': Force-directed layout
        - 'manual': Use resolution as y-coordinate, optimize x-coordinates
        
    show_labels : bool, default=True
        Whether to show cluster labels on nodes.
        
    edge_alpha : float, default=0.6
        Transparency of edges connecting clusters.
        
    edge_style : str, default='bezier'
        Style of edges connecting clusters:
        - 'bezier': Curved bezier edges for natural tree appearance
        - 'straight': Straight lines between nodes
        
    colorbar : bool, default=True
        Whether to show colorbar for continuous color values.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the hierarchical plot.
        
    Examples
    --------
    >>> # Basic usage with just labels (default coloring/sizing)
    >>> labels = {
    ...     0: np.array([0, 0, 1, 1, 2]),
    ...     1: np.array([0, 1, 1, 2, 2])
    ... }
    >>> fig = plot_multiresolution_hierarchy(labels)
    >>> 
    >>> # With custom colors and sizes
    >>> colors = {
    ...     (0, 0): 'red', (0, 1): 'blue', (0, 2): 'green',
    ...     (1, 0): 'red', (1, 1): 'blue', (1, 2): 'yellow'
    ... }
    >>> sizes = {
    ...     (0, 0): 100, (0, 1): 80, (0, 2): 60,
    ...     (1, 0): 120, (1, 1): 90, (1, 2): 70
    ... }
    >>> fig = plot_multiresolution_hierarchy(labels, colors, sizes)
    >>> 
    >>> # With continuous color values
    >>> continuous_colors = {
    ...     (0, 0): 0.1, (0, 1): 0.5, (0, 2): 0.9,
    ...     (1, 0): 0.2, (1, 1): 0.6, (1, 2): 0.8
    ... }
    >>> fig = plot_multiresolution_hierarchy(labels, continuous_colors)
    """
    # Validate inputs
    if not labels_dict:
        raise ValueError("labels_dict cannot be empty")
    
    # Filter out None labels
    valid_labels = {k: v for k, v in labels_dict.items() if v is not None}
    if not valid_labels:
        raise ValueError("No valid labels found in labels_dict")
        
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract cluster information across resolutions
    hierarchy_data = _extract_simple_hierarchy_data(valid_labels, min_overlap)
    
    if not hierarchy_data['nodes']:
        warnings.warn("No clusters found or insufficient overlap between resolutions.")
        ax.text(0.5, 0.5, 'No hierarchy data available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Compute node positions
    positions = _compute_hierarchy_layout(
        hierarchy_data, 
        layout_algorithm=layout_algorithm
    )
    
    # Prepare node properties from provided dictionaries
    node_colors, node_sizes, is_discrete = _prepare_simple_node_properties(
        hierarchy_data,
        color_dict=color_dict,
        size_dict=size_dict, 
        size_range=node_size_range
    )
        
    # Plot edges (connections between resolutions)
    _plot_hierarchy_edges(ax, hierarchy_data, positions, alpha=edge_alpha, style=edge_style)
    
    # Plot nodes (clusters)
    scatter = _plot_hierarchy_nodes(
        ax, positions, node_colors, node_sizes, 
        show_labels=show_labels, hierarchy_data=hierarchy_data,
        is_discrete=is_discrete, cmap=cmap
    )
    
    # Add resolution labels and grid
    _add_simple_resolution_labels(ax, hierarchy_data, positions)
    
    # Configure axes and appearance
    ax.set_ylabel('Resolution Index', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set axis limits with proper padding for all nodes
    if positions:
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding (20% of range, minimum 1.0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_padding = max(x_range * 0.2, 1.0)
        y_padding = max(y_range * 0.1, 0.5)
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        # Fallback if no positions
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
    
    # Add colorbar if requested and data is continuous
    if colorbar and len(set(node_colors)) > 1 and not is_discrete:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Node Color Value', fontsize=11)
    
    # Add legend for node sizes
    _add_simple_size_legend(ax, node_sizes, size_dict, node_size_range)
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_frame_on(False)
    plt.tight_layout()
    return fig

def _extract_simple_hierarchy_data(labels_dict, min_overlap: float) -> Dict:
    """Extract hierarchical cluster data from labels dictionary."""
    
    # Get sorted resolution indices  
    resolution_indices = sorted(labels_dict.keys())
    
    # Build cluster information for each resolution using indices
    nodes = {}  # {(resolution_index, cluster_id): cluster_info}
    edges = []  # [(node1, node2, overlap_score), ...]
    
    for idx in resolution_indices:
        labels = labels_dict[idx]
        if labels is None:
            continue
            
        # Get cluster information
        for cluster_id in np.unique(labels):
            cluster_members = set(np.where(labels == cluster_id)[0])
            cluster_size = len(cluster_members)
            
            node_key = (idx, cluster_id)
            nodes[node_key] = {
                'resolution': idx,  # Use index instead of actual resolution value
                'cluster_id': cluster_id,
                'members': cluster_members,
                'size': cluster_size
            }
    
    # Find edges (connections between adjacent resolutions)
    for i in range(len(resolution_indices) - 1):
        idx_current = resolution_indices[i]
        idx_next = resolution_indices[i + 1]
        
        # Get nodes at current and next resolution
        current_nodes = [key for key in nodes.keys() if key[0] == idx_current]
        next_nodes = [key for key in nodes.keys() if key[0] == idx_next]
        
        # Compute overlaps between all pairs
        for current_node in current_nodes:
            current_members = nodes[current_node]['members']
            
            for next_node in next_nodes:
                next_members = nodes[next_node]['members']
                
                # Compute Jaccard similarity
                if len(current_members) > 0 or len(next_members) > 0:
                    intersection = len(current_members.intersection(next_members))
                    union = len(current_members.union(next_members))
                    jaccard = intersection / union if union > 0 else 0.0
                    
                    if jaccard >= min_overlap:
                        edges.append((current_node, next_node, jaccard))
    
    return {
        'nodes': nodes,
        'edges': edges,
        'resolutions': resolution_indices
    }


def _prepare_simple_node_properties(hierarchy_data: Dict, color_dict: Dict, size_dict: Dict, size_range: Tuple) -> Tuple[List, List, bool]:
    """Prepare node colors and sizes from provided dictionaries."""
    
    nodes = hierarchy_data['nodes']
    
    # Extract color values
    color_values = []
    is_discrete = False
    
    for node_key in nodes:
        if color_dict and node_key in color_dict:
            color_val = color_dict[node_key]
            color_values.append(color_val)
        else:
            # Default to cluster labels (discrete coloring)
            color_values.append(nodes[node_key]['cluster_id'])
            is_discrete = True
    
    # Check if colors are discrete (strings or integers) - override if color_dict is None
    if color_dict is None:
        is_discrete = True
    elif color_values:
        is_discrete = any(isinstance(val, (str, int, np.integer)) for val in color_values)
    
    # Extract size values
    size_values = []
    for node_key in nodes:
        if size_dict and node_key in size_dict:
            size_val = size_dict[node_key]
            size_values.append(size_val)
        else:
            # Default to cluster size
            size_values.append(nodes[node_key]['size'])
    
    # Normalize sizes to range
    if size_values and max(size_values) > min(size_values):
        min_val, max_val = min(size_values), max(size_values)
        normalized_sizes = [
            size_range[0] + (val - min_val) / (max_val - min_val) * (size_range[1] - size_range[0])
            for val in size_values
        ]
    else:
        normalized_sizes = [size_range[0]] * len(size_values)
    
    return color_values, normalized_sizes, is_discrete


def _add_simple_resolution_labels(ax: plt.Axes, hierarchy_data: Dict, positions: Dict):
    """Add resolution index labels to y-axis."""
    
    resolutions = hierarchy_data['resolutions']
    
    # Get unique y-coordinates and corresponding resolutions
    y_positions = {}
    for node_key, pos in positions.items():
        res = node_key[0]
        y_positions[pos[1]] = res
    
    # Set y-tick labels  
    y_ticks = sorted(y_positions.keys())
    y_labels = [f"{y_positions[y]}" for y in y_ticks]
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)


def _add_simple_size_legend(ax: plt.Axes, sizes: List, size_dict: Dict, size_range: Tuple):
    """Add legend explaining node sizes."""
    
    if not sizes or len(set(sizes)) <= 1:
        return
    
    # Create size legend with a few representative sizes
    size_values = [min(sizes), np.median(sizes), max(sizes)]
    
    # Try to determine what sizes represent
    if size_dict:
        size_label = "Custom Size"
    else:
        size_label = "Cluster Size"
    
    size_labels = [f"{size_label}: {val:.1f}" for val in size_values]
    
    # Create legend elements
    legend_elements = []
    for i, (size_val, label) in enumerate(zip(size_values, size_labels)):
        # Map size value to actual node size
        if max(sizes) > min(sizes):
            normalized_size = size_range[0] + (size_val - min(sizes)) / (max(sizes) - min(sizes)) * (size_range[1] - size_range[0])
        else:
            normalized_size = size_range[0]
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=np.sqrt(normalized_size/10), label=label, markeredgecolor='black')
        )
    
    # Add legend to plot
    legend = ax.legend(handles=legend_elements, loc=(1.01,0), 
                      title=f"Node Size", framealpha=0.9)
    legend.get_title().set_fontsize(10)


def _extract_hierarchy_data(multiresolution_results, min_overlap: float) -> Dict:
    """Extract hierarchical cluster data from multiresolution results."""
    
    # Get sorted resolution indices (we'll use indices for positioning, not actual values)
    resolution_indices = sorted(multiresolution_results.labels_.keys())
    
    # Build cluster information for each resolution using indices
    nodes = {}  # {(resolution_index, cluster_id): cluster_info}
    edges = []  # [(node1, node2, overlap_score), ...]
    
    for idx in resolution_indices:
        labels = multiresolution_results.labels_[idx]
        if labels is None:
            continue
            
        # Get cluster information
        for cluster_id in np.unique(labels):
            cluster_members = set(np.where(labels == cluster_id)[0])
            cluster_size = len(cluster_members)
            
            # Get consistency info if available
            consistency_info = None
            if (hasattr(multiresolution_results, 'consistency_summaries_') and 
                idx in multiresolution_results.consistency_summaries_ and
                multiresolution_results.consistency_summaries_[idx] and
                'per_cluster_consistency' in multiresolution_results.consistency_summaries_[idx]):
                
                per_cluster = multiresolution_results.consistency_summaries_[idx]['per_cluster_consistency']
                if per_cluster:
                    # Find matching cluster in consistency data
                    for track_id, track_info in per_cluster.items():
                        # Simple heuristic: match by size similarity
                        if abs(track_info['mean_size'] - cluster_size) < cluster_size * 0.1:
                            consistency_info = track_info
                            break
            
            node_key = (idx, cluster_id)
            nodes[node_key] = {
                'resolution': idx,  # Now using index instead of actual resolution value
                'cluster_id': cluster_id,
                'members': cluster_members,
                'size': cluster_size,
                'consistency': consistency_info['mean_consistency'] if consistency_info else None,
                'stability': consistency_info['stability'] if consistency_info else None
            }
    
    # Find edges (connections between adjacent resolutions)
    for i in range(len(resolution_indices) - 1):
        idx_current = resolution_indices[i]
        idx_next = resolution_indices[i + 1]
        
        # Get nodes at current and next resolution
        current_nodes = [key for key in nodes.keys() if key[0] == idx_current]
        next_nodes = [key for key in nodes.keys() if key[0] == idx_next]
        
        # Compute overlaps between all pairs
        for current_node in current_nodes:
            current_members = nodes[current_node]['members']
            
            for next_node in next_nodes:
                next_members = nodes[next_node]['members']
                
                # Compute Jaccard similarity
                if len(current_members) > 0 or len(next_members) > 0:
                    intersection = len(current_members.intersection(next_members))
                    union = len(current_members.union(next_members))
                    jaccard = intersection / union if union > 0 else 0.0
                    
                    if jaccard >= min_overlap:
                        edges.append((current_node, next_node, jaccard))
    
    return {
        'nodes': nodes,
        'edges': edges,
        'resolutions': resolution_indices  # Return indices instead of actual resolution values
    }


def _compute_hierarchy_layout(hierarchy_data: Dict, layout_algorithm: str = 'tree') -> Dict:
    """Compute node positions for hierarchical layout."""
    
    nodes = hierarchy_data['nodes']
    edges = hierarchy_data['edges']
    resolutions = hierarchy_data['resolutions']
    
    positions = {}
    
    if layout_algorithm == 'organic' or layout_algorithm == 'tree':
        # Organic tree layout: splits emanate from parent centers
        positions = _compute_organic_tree_layout(hierarchy_data)
        
        # Apply spacing adjustments for organic layout
        if layout_algorithm == 'organic':
            positions = _add_organic_layout_spacing(positions)
        
    elif layout_algorithm == 'manual':
        # Use resolution as y-coordinate, optimize x-coordinates to minimize crossings
        
        # Group nodes by resolution
        res_nodes = {}
        for node_key in nodes:
            res = node_key[0]
            if res not in res_nodes:
                res_nodes[res] = []
            res_nodes[res].append(node_key)
        
        # Sort resolutions and assign y-coordinates
        sorted_resolutions = sorted(resolutions)
        
        for i, res in enumerate(sorted_resolutions):
            y_coord = i
            
            if res not in res_nodes:
                continue
                
            # Sort nodes at this resolution by size (largest first)
            res_nodes_list = sorted(
                res_nodes[res], 
                key=lambda x: nodes[x]['size'], 
                reverse=True
            )
            
            # Simple horizontal spacing
            for j, node_key in enumerate(res_nodes_list):
                x_coord = j
                positions[node_key] = (x_coord, y_coord)
        
        # Optimize x-coordinates to minimize edge crossings
        positions = _optimize_x_positions(positions, edges, res_nodes, sorted_resolutions)
    
    elif layout_algorithm == 'spring' and HAS_NETWORKX:
        # Use NetworkX spring layout
        G = nx.Graph()
        
        # Add nodes
        for node_key in nodes:
            G.add_node(node_key)
        
        # Add edges
        for edge in edges:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        
        # Compute spring layout
        nx_positions = nx.spring_layout(G, k=2.0, iterations=50)
        positions = {node: (pos[0], pos[1]) for node, pos in nx_positions.items()}
    
    elif layout_algorithm == 'igraph_tree' and HAS_IGRAPH:
        # Use igraph tree layout for hierarchical structure
        positions = _compute_igraph_tree_layout(hierarchy_data)
    
    else:
        # Fallback: simple grid layout
        for i, node_key in enumerate(nodes.keys()):
            positions[node_key] = (i % 5, i // 5)
    
    return positions


def _compute_organic_tree_layout(hierarchy_data: Dict) -> Dict:
    """
    Compute organic tree layout where clusters split from parent centers.
    
    Creates a natural tree structure flowing from top to bottom with splits
    emanating from parent positions rather than simple grid positioning.
    """
    
    nodes = hierarchy_data['nodes']
    edges = hierarchy_data['edges']
    resolutions = sorted(hierarchy_data['resolutions'])
    
    positions = {}
    
    # Build parent-child relationships
    parent_child_map = _build_parent_child_relationships(edges, resolutions)
    
    # Group nodes by resolution level
    level_nodes = {}
    for node_key in nodes:
        level = resolutions.index(node_key[0])
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node_key)
    
    # Position root level (highest resolution = top of tree)
    root_level = 0
    if root_level in level_nodes:
        root_nodes = sorted(level_nodes[root_level], key=lambda x: nodes[x]['size'], reverse=True)
        
        # Arrange root nodes in a horizontal line
        total_width = len(root_nodes) * 2  # Space between root nodes
        start_x = -(total_width - 1) / 2
        
        for i, node in enumerate(root_nodes):
            x_pos = start_x + i * 2
            y_pos = len(resolutions) - 1  # Top of tree
            positions[node] = (x_pos, y_pos)
    
    # Position subsequent levels using parent-based splitting
    for level in range(1, len(resolutions)):
        if level not in level_nodes:
            continue
            
        y_pos = len(resolutions) - 1 - level  # Decrease y as we go down
        
        # Group children by their parents
        parent_groups = {}
        orphan_nodes = []
        
        for node in level_nodes[level]:
            parents = parent_child_map.get(node, [])
            if parents:
                # Use the strongest parent (highest overlap)
                strongest_parent = max(parents, key=lambda x: x[1])
                parent_node = strongest_parent[0]
                
                if parent_node not in parent_groups:
                    parent_groups[parent_node] = []
                parent_groups[parent_node].append(node)
            else:
                orphan_nodes.append(node)
        
        # Position children around their parents
        for parent_node, children in parent_groups.items():
            if parent_node not in positions:
                continue
                
            parent_x = positions[parent_node][0]
            n_children = len(children)
            
            if n_children == 1:
                # Single child: position directly below parent
                positions[children[0]] = (parent_x, y_pos)
                
            else:
                # Multiple children: spread symmetrically around parent
                children_sorted = sorted(children, key=lambda x: nodes[x]['size'], reverse=True)
                
                # Calculate spread width based on number of children (more conservative)
                base_spread = min(n_children * 0.6, 3.0)  # Reduced max spread to 3.0
                
                # Adjust spread based on tree depth to prevent excessive width
                current_level = level  # Use the current level variable
                depth_factor = 1.0 - (current_level * 0.1)  # Reduce spread at deeper levels
                spread_width = base_spread * max(depth_factor, 0.5)
                
                if n_children % 2 == 1:
                    # Odd number: center child at parent position
                    center_idx = n_children // 2
                    positions[children_sorted[center_idx]] = (parent_x, y_pos)
                    
                    # Position others symmetrically
                    if n_children > 1:
                        step_size = spread_width / (n_children - 1)
                        
                        for i, child in enumerate(children_sorted):
                            if i == center_idx:
                                continue
                            
                            if i < center_idx:
                                offset = -(center_idx - i) * step_size
                            else:
                                offset = (i - center_idx) * step_size
                            
                            positions[child] = (parent_x + offset, y_pos)
                    
                else:
                    # Even number: position symmetrically around parent
                    step_size = spread_width / max(n_children, 1)
                    start_offset = -(n_children - 1) * step_size / 2
                    
                    for i, child in enumerate(children_sorted):
                        offset = start_offset + i * step_size
                        positions[child] = (parent_x + offset, y_pos)
        
        # Handle orphan nodes (no clear parent)
        if orphan_nodes:
            # Find empty space and place orphans there
            existing_x = [pos[0] for pos in positions.values() if pos[1] == y_pos]
            
            if existing_x:
                # Place orphans to the right of existing nodes
                start_x = max(existing_x) + 2
            else:
                # No existing nodes at this level
                start_x = 0
            
            for i, orphan in enumerate(orphan_nodes):
                positions[orphan] = (start_x + i * 1.5, y_pos)
    
    return positions

def _build_parent_child_relationships(edges: List, resolutions: List) -> Dict:
    """
    Build parent-child relationships from edge list.
    
    Returns:
    --------
    parent_child_map : dict
        Maps child nodes to list of (parent_node, overlap_weight) tuples
    """
    
    parent_child_map = {}
    
    for edge in edges:
        parent_node, child_node, overlap_weight = edge
        parent_res = parent_node[0]
        child_res = child_node[0]
        
        # Ensure parent is at earlier resolution (higher in tree)
        if resolutions.index(parent_res) < resolutions.index(child_res):
            # parent_node is actually the parent
            if child_node not in parent_child_map:
                parent_child_map[child_node] = []
            parent_child_map[child_node].append((parent_node, overlap_weight))
        else:
            # child_node is actually the parent (reverse relationship)
            if parent_node not in parent_child_map:
                parent_child_map[parent_node] = []
            parent_child_map[parent_node].append((child_node, overlap_weight))
    
    return parent_child_map


def _add_organic_layout_spacing(positions: Dict) -> Dict:
    """
    Add additional spacing adjustments to prevent node overlap in organic layout.
    """
    
    adjusted_positions = positions.copy()
    
    # Get all y-levels
    y_levels = {}
    for node, (x, y) in positions.items():
        if y not in y_levels:
            y_levels[y] = []
        y_levels[y].append((node, x))
    
    # For each level, ensure minimum spacing between nodes
    min_spacing = 0.5
    
    for y, nodes_at_level in y_levels.items():
        if len(nodes_at_level) <= 1:
            continue
            
        # Sort by x-position
        nodes_at_level.sort(key=lambda x: x[1])
        
        # Adjust positions to maintain minimum spacing
        current_x = nodes_at_level[0][1]
        adjusted_positions[nodes_at_level[0][0]] = (current_x, y)
        
        for i in range(1, len(nodes_at_level)):
            node, original_x = nodes_at_level[i]
            required_x = current_x + min_spacing
            
            if original_x < required_x:
                # Need to shift right
                adjusted_positions[node] = (required_x, y)
                current_x = required_x
            else:
                # Original position is fine
                adjusted_positions[node] = (original_x, y)
                current_x = original_x
    
    return adjusted_positions


def _compute_igraph_tree_layout(hierarchy_data: Dict) -> Dict:
    """
    Compute node positions using igraph's tree layout algorithms.
    
    This creates a hierarchical tree layout that minimizes edge crossings
    by using igraph's optimized tree layout algorithms.
    """
    
    nodes = hierarchy_data['nodes']
    edges = hierarchy_data['edges']
    resolutions = sorted(hierarchy_data['resolutions'])
    
    # Build parent-child relationships and create tree structure
    parent_child_map = _build_parent_child_relationships(edges, resolutions)
    
    # Create a tree by keeping only the strongest parent-child connections
    tree_edges = []
    tree_edge_weights = []
    
    # For each child, keep only the strongest parent connection
    for child_node, parents in parent_child_map.items():
        if parents:
            # Find parent with highest overlap weight
            strongest_parent, weight = max(parents, key=lambda x: x[1])
            tree_edges.append((strongest_parent, child_node))
            tree_edge_weights.append(weight)
    
    # Create igraph graph
    g = ig.Graph()
    
    # Add all nodes
    node_list = list(nodes.keys())
    g.add_vertices(len(node_list))
    
    # Create mapping from node keys to vertex indices
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Add tree edges with igraph vertex indices
    igraph_edges = []
    for parent, child in tree_edges:
        if parent in node_to_idx and child in node_to_idx:
            igraph_edges.append((node_to_idx[parent], node_to_idx[child]))
    
    if igraph_edges:
        g.add_edges(igraph_edges)
    
    # Find root nodes (nodes with no parents in tree)
    root_candidates = []
    children_in_tree = set(child for _, child in tree_edges)
    for node in node_list:
        if node not in children_in_tree:
            root_candidates.append(node_to_idx[node])
    
    # Use the largest root as the main root, or first node if no clear root
    if root_candidates:
        # Choose root by cluster size (largest first)
        root_idx = max(root_candidates, key=lambda idx: nodes[idx_to_node[idx]]['size'])
    else:
        # Fallback to first node
        root_idx = 0
    
    # Apply igraph tree layout
    try:
        # Try Reingold-Tilford tree layout (good for hierarchical data)
        layout = g.layout_reingold_tilford(mode="out", root=[root_idx])
    except:
        try:
            # Fallback to simpler tree layout
            layout = g.layout_tree(mode="out", root=[root_idx])
        except:
            # Final fallback to fruchterman-reingold
            layout = g.layout_fruchterman_reingold()
    
    # Convert layout back to node positions
    positions = {}
    for idx, (x, y) in enumerate(layout.coords):
        node_key = idx_to_node[idx]
        # Flip y-axis so root is at top
        positions[node_key] = (x, -y)
    
    return positions


def _optimize_x_positions(positions: Dict, edges: List, res_nodes: Dict, resolutions: List) -> Dict:
    """Optimize x-positions to minimize edge crossings using a layer-by-layer approach."""
    
    optimized_positions = positions.copy()
    
    # Build adjacency information for easier access
    edge_dict = {}
    for edge in edges:
        node1, node2, weight = edge
        if node1 not in edge_dict:
            edge_dict[node1] = []
        if node2 not in edge_dict:
            edge_dict[node2] = []
        edge_dict[node1].append((node2, weight))
        edge_dict[node2].append((node1, weight))
    
    # Process each resolution level (starting from second level)
    for i in range(1, len(resolutions)):
        current_res = resolutions[i]
        prev_res = resolutions[i-1]
        
        if current_res not in res_nodes or prev_res not in res_nodes:
            continue
        
        current_nodes = res_nodes[current_res]
        prev_nodes = res_nodes[prev_res]
        
        if not current_nodes or not prev_nodes:
            continue
        
        # Calculate optimal ordering for current level nodes
        optimal_order = _calculate_optimal_ordering(
            current_nodes, prev_nodes, edge_dict, optimized_positions
        )
        
        # Update x-positions based on optimal ordering
        y_coord = optimized_positions[current_nodes[0]][1]
        for j, node in enumerate(optimal_order):
            optimized_positions[node] = (j, y_coord)
    
    return optimized_positions


def _calculate_optimal_ordering(current_nodes: List, prev_nodes: List, 
                               edge_dict: Dict, positions: Dict) -> List:
    """
    Calculate optimal ordering of nodes to minimize crossings.
    
    Uses a barycenter heuristic followed by local optimization.
    """
    
    # Calculate barycenter (weighted average) positions for each current node
    node_barycenters = []
    
    for node in current_nodes:
        connected_prev_positions = []
        total_weight = 0
        
        # Find connections to previous level
        if node in edge_dict:
            for connected_node, weight in edge_dict[node]:
                if connected_node in positions and connected_node in prev_nodes:
                    prev_x = positions[connected_node][0]
                    connected_prev_positions.append(prev_x * weight)
                    total_weight += weight
        
        # Calculate barycenter
        if total_weight > 0:
            barycenter = sum(connected_prev_positions) / total_weight
        else:
            # For nodes with no connections, use node index as fallback
            barycenter = current_nodes.index(node)
        
        node_barycenters.append((node, barycenter))
    
    # Sort nodes by barycenter
    node_barycenters.sort(key=lambda x: x[1])
    initial_order = [node for node, _ in node_barycenters]
    
    # Apply local optimization to reduce crossings further
    optimized_order = _local_crossing_optimization(
        initial_order, prev_nodes, edge_dict, positions
    )
    
    return optimized_order


def _local_crossing_optimization(node_order: List, prev_nodes: List, 
                                edge_dict: Dict, positions: Dict, max_iterations: int = 10) -> List:
    """
    Apply local search to minimize crossings by swapping adjacent nodes.
    """
    
    current_order = node_order.copy()
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try swapping adjacent pairs
        for i in range(len(current_order) - 1):
            # Calculate crossings before swap
            crossings_before = _count_crossings_for_pair(
                current_order, i, i+1, prev_nodes, edge_dict, positions
            )
            
            # Swap and calculate crossings after
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            crossings_after = _count_crossings_for_pair(
                current_order, i, i+1, prev_nodes, edge_dict, positions
            )
            
            # Keep swap if it reduces crossings
            if crossings_after < crossings_before:
                improved = True
            else:
                # Revert swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
        
        # Stop if no improvement
        if not improved:
            break
    
    return current_order


def _count_crossings_for_pair(node_order: List, idx1: int, idx2: int, 
                             prev_nodes: List, edge_dict: Dict, positions: Dict) -> int:
    """
    Count crossings caused by edges from two specific nodes.
    """
    
    node1 = node_order[idx1]
    node2 = node_order[idx2]
    
    crossings = 0
    
    # Get connections for both nodes
    connections1 = []
    connections2 = []
    
    if node1 in edge_dict:
        for connected_node, weight in edge_dict[node1]:
            if connected_node in prev_nodes and connected_node in positions:
                connections1.append(positions[connected_node][0])
    
    if node2 in edge_dict:
        for connected_node, weight in edge_dict[node2]:
            if connected_node in prev_nodes and connected_node in positions:
                connections2.append(positions[connected_node][0])
    
    # Count crossings between edges from node1 and node2
    for pos1 in connections1:
        for pos2 in connections2:
            # Check if edges cross (considering node1 is to the left of node2)
            if idx1 < idx2 and pos1 > pos2:  # node1 left, but connects to right
                crossings += 1
            elif idx1 > idx2 and pos1 < pos2:  # node1 right, but connects to left
                crossings += 1
    
    return crossings


def _calculate_total_crossings(node_order: List, prev_nodes: List, 
                              edge_dict: Dict, positions: Dict) -> int:
    """
    Calculate total number of edge crossings for a given node ordering.
    """
    
    total_crossings = 0
    
    # Check all pairs of nodes in current level
    for i in range(len(node_order)):
        for j in range(i + 1, len(node_order)):
            crossings = _count_crossings_for_pair(
                node_order, i, j, prev_nodes, edge_dict, positions
            )
            total_crossings += crossings
    
    return total_crossings


def _prepare_node_properties(hierarchy_data: Dict, color_by: str, size_by: str, size_range: Tuple) -> Tuple[List, List, bool]:
    """Prepare node colors and sizes based on specified properties."""
    
    nodes = hierarchy_data['nodes']
    
    # Extract color values
    color_values = []
    is_discrete = False  # Flag to indicate if colors should be treated as discrete
    
    for node_key in nodes:
        node_info = nodes[node_key]
        
        if color_by == 'size':
            color_values.append(node_info['size'])
        elif color_by == 'consistency':
            consistency = node_info['consistency']
            color_values.append(consistency if consistency is not None else 0.0)
        elif color_by == 'stability':
            stability = node_info['stability']
            color_values.append(stability if stability is not None else 0.0)
        elif color_by == 'cluster':
            # Color by cluster label/ID order - this should be discrete
            color_values.append(node_info['cluster_id'])
            is_discrete = True
        else:
            color_values.append(0.0)
    
    # Check if all values are integers (indicating categorical/discrete data)
    if not is_discrete and color_values:
        is_discrete = all(isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()) 
                         for val in color_values if val is not None)
    
    # Extract size values
    size_values = []
    for node_key in nodes:
        node_info = nodes[node_key]
        
        if size_by == 'size':
            size_values.append(node_info['size'])
        elif size_by == 'consistency':
            consistency = node_info['consistency']
            size_values.append(consistency if consistency is not None else 0.0)
        elif size_by == 'stability':
            stability = node_info['stability']
            size_values.append(stability if stability is not None else 0.0)
        else:
            size_values.append(1.0)
    
    # Normalize sizes to range
    if size_values and max(size_values) > min(size_values):
        min_val, max_val = min(size_values), max(size_values)
        normalized_sizes = [
            size_range[0] + (val - min_val) / (max_val - min_val) * (size_range[1] - size_range[0])
            for val in size_values
        ]
    else:
        normalized_sizes = [size_range[0]] * len(size_values)
    
    return color_values, normalized_sizes, is_discrete


def _plot_hierarchy_edges(ax: plt.Axes, hierarchy_data: Dict, positions: Dict, alpha: float, style: str = 'bezier'):
    """Plot edges connecting clusters between resolutions."""
    
    edges = hierarchy_data['edges']
    
    if style == 'bezier':
        # Plot curved bezier edges
        for edge in edges:
            node1, node2, weight = edge
            if node1 in positions and node2 in positions:
                pos1 = positions[node1]
                pos2 = positions[node2]
                
                # Create bezier curve
                bezier_path = _create_bezier_edge(pos1, pos2)
                
                # Plot the bezier curve
                line_width = weight * 2  # Scale line width by overlap weight
                ax.plot(bezier_path[:, 0], bezier_path[:, 1], 
                       color='gray', alpha=alpha, linewidth=line_width, 
                       solid_capstyle='round')
    
    else:  # straight lines
        # Create edge lines
        lines = []
        edge_weights = []
        
        for edge in edges:
            node1, node2, weight = edge
            if node1 in positions and node2 in positions:
                pos1 = positions[node1]
                pos2 = positions[node2]
                lines.append([pos1, pos2])
                edge_weights.append(weight * 2)  # Scale for visibility
        
        if lines:
            # Create line collection
            lc = LineCollection(lines, alpha=alpha, linewidths=edge_weights)
            lc.set_color('gray')
            ax.add_collection(lc)


def _create_bezier_edge(start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                       curvature: float = 0.8, num_points: int = 50) -> np.ndarray:
    """
    Create a natural tree-like bezier curve that exits vertically from parent,
    curves horizontally, and enters vertically to child.
    
    Parameters
    ----------
    start_pos : tuple
        Starting position (x, y) - parent node
    end_pos : tuple
        Ending position (x, y) - child node
    curvature : float, default=0.8
        Curvature strength (0 = straight line, higher = more curved)
    num_points : int, default=50
        Number of points in the curve
        
    Returns
    -------
    curve_points : np.ndarray
        Array of (x, y) points defining the natural tree bezier curve
    """
    
    x1, y1 = start_pos  # Parent node (higher in tree)
    x2, y2 = end_pos    # Child node (lower in tree)
    
    # Calculate distances for adaptive control point placement
    vertical_distance = abs(y2 - y1)
    horizontal_distance = abs(x2 - x1)
    total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
    
    # If points are too close, return straight line
    if total_distance < 0.05:
        t = np.linspace(0, 1, num_points)
        curve_points = np.column_stack([
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1)
        ])
        return curve_points
    
    # ADAPTIVE CURVATURE CALCULATION
    # Base vertical offset scales with distance and curvature
    # For short distances: more pronounced curves (higher factor)
    # For long distances: moderate curves (lower factor)
    distance_factor = min(1.0 / (total_distance + 0.1), 2.0)  # Inverse relationship with distance
    base_vertical_offset = vertical_distance * curvature * (0.3 + 0.4 * distance_factor)
    
    # Adaptive horizontal offset based on both distances
    horizontal_ratio = horizontal_distance / (vertical_distance + 0.1)  # Prevent division by zero
    horizontal_curve_strength = min(horizontal_ratio * 0.5, 1.0) * curvature
    
    # CONTROL POINTS FOR PURE VERTICAL ENTRY/EXIT
    # Use a different strategy: create control points that force vertical tangents
    
    # For pure vertical entry/exit, the control points must be positioned such that:
    # 1. First control point is directly below parent (same x)
    # 2. Second control point is directly above child (same x)  
    # 3. The curve will naturally be vertical at start and end
    
    # Calculate control point distances
    control_distance = base_vertical_offset
    
    # First control point: EXACTLY below parent (guarantees vertical exit)
    ctrl1_x = x1  # EXACTLY same X as parent
    ctrl1_y = y1 - control_distance
    
    # Second control point: EXACTLY above child (guarantees vertical entry)
    ctrl2_x = x2  # EXACTLY same X as child
    ctrl2_y = y2 + control_distance
    
    # The horizontal curvature will come naturally from the bezier interpolation
    # between the offset control points - no need to modify X positions
    
    # Generate bezier curve using cubic bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    # P₀ = start (parent), P₁ = ctrl1 (vertical exit), P₂ = ctrl2 (vertical approach), P₃ = end (child)
    t = np.linspace(0, 1, num_points)
    
    # Vectorized bezier calculation
    term1 = (1 - t)**3
    term2 = 3 * (1 - t)**2 * t
    term3 = 3 * (1 - t) * t**2
    term4 = t**3
    
    x = term1 * x1 + term2 * ctrl1_x + term3 * ctrl2_x + term4 * x2
    y = term1 * y1 + term2 * ctrl1_y + term3 * ctrl2_y + term4 * y2
    
    curve_points = np.column_stack([x, y])
    
    return curve_points


def _plot_hierarchy_nodes(ax: plt.Axes, positions: Dict, colors: List, sizes: List, 
                         show_labels: bool, hierarchy_data: Dict, is_discrete: bool = False,
                         cmap: Any = None) -> plt.scatter:
    """Plot cluster nodes in the hierarchy."""
    
    nodes = hierarchy_data['nodes']
    node_keys = list(nodes.keys())
    
    # Extract positions
    x_coords = [positions[key][0] for key in node_keys]
    y_coords = [positions[key][1] for key in node_keys]
    
    # Choose appropriate colormap based on data type
    if cmap is None and colors is not None:
        if len(np.unique(colors)) <= 10:
            cmap = "tab10"
        elif len(np.unique(colors)) <= 20:
            cmap = "tab20"
        else:
            cmap = distinctipy.get_colors(len(np.unique(colors)), pastel_factor=0.3, rng=0)

    scatter = sns.scatterplot(
        x=x_coords, y=y_coords,
        hue=colors,
        hue_order=np.sort(np.unique(colors)) if is_discrete else None,
        palette=cmap,
        size=sizes,
        sizes=(min(sizes), max(sizes)),
        alpha=1, edgecolors='black', linewidth=0.5,
        ax=ax
    )

    # Add labels if requested
    if show_labels:
        for i, node_key in enumerate(node_keys):
            cluster_id = node_key[1]
            cluster_size = nodes[node_key]['size']
            label = f"{cluster_id}\n({cluster_size})"
            
            ax.annotate(label, (x_coords[i], y_coords[i]), 
                       xytext=(15, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    return scatter


def _add_resolution_labels(ax: plt.Axes, hierarchy_data: Dict, positions: Dict):
    """Add resolution index labels to y-axis."""
    
    resolutions = hierarchy_data['resolutions']
    
    # Get unique y-coordinates and corresponding resolutions
    y_positions = {}
    for node_key, pos in positions.items():
        res = node_key[0]
        y_positions[pos[1]] = res
    
    # Set y-tick labels  
    y_ticks = sorted(y_positions.keys())
    y_labels = [f"{y_positions[y]}" for y in y_ticks]
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)


def _add_size_legend(ax: plt.Axes, sizes: List, size_by: str, size_range: Tuple):
    """Add legend explaining node sizes."""
    
    if not sizes or len(set(sizes)) <= 1:
        return
    
    # Create size legend with a few representative sizes
    size_values = [min(sizes), np.median(sizes), max(sizes)]
    size_labels = [f"{size_by.capitalize()}: {val:.1f}" for val in size_values]
    
    # Create legend elements
    legend_elements = []
    for i, (size_val, label) in enumerate(zip(size_values, size_labels)):
        # Map size value to actual node size
        if max(sizes) > min(sizes):
            normalized_size = size_range[0] + (size_val - min(sizes)) / (max(sizes) - min(sizes)) * (size_range[1] - size_range[0])
        else:
            normalized_size = size_range[0]
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=np.sqrt(normalized_size/10), label=label, markeredgecolor='black')
        )
    
    # Add legend to plot
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                      title=f"Node Size ({size_by.capitalize()})", framealpha=0.9)
    legend.get_title().set_fontsize(10)