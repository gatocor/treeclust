"""
Scanpy-style plotting functions for treeclust.

This module provides scanpy-compatible plotting functions that work with
AnnData objects and follow scanpy's plotting conventions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, Sequence, Dict, Any
import warnings
from scipy import sparse
# Cluster rows (groups) based on expression patterns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as mcolors

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
    warnings.warn("AnnData not available. Plotting functions will have limited functionality.")

from ...plotting import (
    plot_graph,
    plot_multiresolution_graph,
    plot_multiresolution_hierarchy,
    plot_clustering_comparison,
    plot_multiresolution_analysis,
    plot_cluster_expression_dotplot
)

def graph(
    adata: 'anndata.AnnData',
    *,
    color: Optional[Union[str, Sequence[str]]] = None,
    use_rep: str = 'X_umap',
    neighbors_key: Optional[str] = 'neighbors',
    adjacency_key: str = 'connectivities',
    show_graph: bool = True,
    alpha: float = 1.0,
    size: Optional[float] = None,
    palette: Optional[str] = None,
    legend_loc: str = 'right margin',
    legend_fontsize: Optional[int] = None,
    save: Optional[str] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Graph plot with optional neighborhood graph overlay using treeclust.
    
    Similar to scanpy.pl.umap but works with any embedding and has enhanced 
    graph visualization capabilities from treeclust for showing neighborhood 
    connections and clustering structure.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    color : str or sequence of str, optional
        Keys for annotations of observations/cells or variables/genes.
        
    use_rep : str, default='X_umap'
        Use the indicated representation from .obsm (e.g., 'X_umap', 'X_pca', 'X_tsne').
        
    neighbors_key : str, optional, default='neighbors'
        Key in adata.uns containing neighborhood information.
        
    adjacency_key : str, default='connectivities'
        Which adjacency matrix to use ('connectivities', 'distances').
        
    show_graph : bool, default=True
        Whether to overlay the neighborhood graph on the embedding.
        
    alpha : float, default=1.0
        Alpha value for points.
        
    size : float, optional
        Point size. If None, automatically determined.
        
    palette : str, optional
        Color palette to use.
        
    legend_loc : str, default='right margin'
        Location of legend.
        
    legend_fontsize : int, optional
        Legend font size.
        
    save : str, optional
        Save figure to file.
        
    **kwargs
        Additional arguments passed to plot_graph.
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object if save is None.
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> # Basic graph plot using UMAP coordinates
    >>> tcsc.pl.graph(adata, color='leiden')
    >>> # Graph plot with neighborhood overlay
    >>> tcsc.pl.graph(adata, color='leiden', show_graph=True)
    >>> # Use different embedding (e.g., PCA)
    >>> tcsc.pl.graph(adata, color='leiden', use_rep='X_pca')
    """
    if not HAS_ANNDATA:
        raise ImportError("AnnData is required for plotting functions")
    
    if use_rep not in adata.obsm:
        raise ValueError(f"Representation '{use_rep}' not found in adata.obsm")
    
    # Extract coordinates
    coords = adata.obsm[use_rep]
    
    # Handle color parameter
    if color is not None:
        if isinstance(color, str):
            color_data = adata.obs[color] if color in adata.obs else None
        else:
            # Multiple colors - for now just use the first one
            color_data = adata.obs[color[0]] if color[0] in adata.obs else None
    else:
        color_data = None
    
    # Extract adjacency matrix for graph overlay
    adjacency = None
    if show_graph:
        # First try standard scanpy location in adata.obsp
        if adjacency_key in adata.obsp:
            adjacency = adata.obsp[adjacency_key]
        # Then try treeclust-specific keys in adata.obsp
        elif f'{neighbors_key}_{adjacency_key}' in adata.obsp:
            adjacency = adata.obsp[f'{neighbors_key}_{adjacency_key}']
        # Finally try the old location in adata.uns[neighbors_key]
        elif neighbors_key in adata.uns:
            neighbors_info = adata.uns[neighbors_key]
            if adjacency_key in neighbors_info:
                adjacency = neighbors_info[adjacency_key]
    
    # Use treeclust's plot_graph function
    plot_kwargs = {}
    
    # Add edge parameters if showing graph
    if adjacency is not None and show_graph:
        plot_kwargs['edge_alpha'] = 0.3
        edge_matrix = adjacency
    else:
        # Create empty sparse matrix for no edges
        n_nodes = coords.shape[0]
        edge_matrix = sparse.csr_matrix((n_nodes, n_nodes))
        plot_kwargs['edge_alpha'] = 0.0  # Make edges invisible
    
    fig = plot_graph(
        edge_width=edge_matrix,
        node_pos=coords,
        node_color=color_data,
        node_size=size if size is not None else 50,
        **plot_kwargs
    )
    
    return fig

def multiresolution_leiden(
    adata: 'anndata.AnnData',
    *,
    key: str = 'multiresolution_leiden',
    coordinates_key: str = 'X_umap',
    resolutions: Optional[Sequence[float]] = None,
    confidence_key: Optional[str] = None,
    neighbors_key: Optional[str] = 'neighbors',
    adjacency_key: str = 'connectivities',
    show_graph: bool = False,
    show_cluster_labels: bool = True,
    palette: Optional[str] = None,
    save: Optional[str] = None,
    skip_repeated: bool = True,
    figsize: Optional[tuple] = None,
    n_cols: Optional[tuple] = 3,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Plot multiresolution Leiden clustering results using treeclust.
    
    Visualizes clustering results across multiple resolutions in a grid layout,
    showing how cluster structure changes with resolution parameter.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    key : str, default='multiresolution_leiden'
        Key in adata.obs containing multiresolution clustering results.
        
    coordinates_key : str, default='X_umap'
        Key in adata.obsm for coordinates to plot.
        
    resolutions : sequence of float, optional
        Specific resolutions to plot. If None, plots all available.
        
    confidence_key : str, optional
        Key for confidence/stability information to show as node sizes.
        
    neighbors_key : str, default='neighbors'
        Key in adata.uns or adata.obsp for neighbor information.
        
    adjacency_key : str, default='connectivities'
        Key for adjacency/connectivity matrix within neighbors data.
        
    show_graph : bool, default=False
        Whether to show edges from the neighbor graph.
        
    show_cluster_labels : bool, default=True
        Whether to show cluster labels on the plots.
        
    palette : str, optional
        Color palette for clusters.
        
    save : str, optional
        Save figure to file.
        
    **kwargs
        Additional arguments passed to plot_graph.
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object if save is None.
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> tcsc.pl.multiresolution_leiden(adata, key='multires_leiden')
    """
    if not HAS_ANNDATA:
        raise ImportError("AnnData is required for plotting functions")
    
    if coordinates_key not in adata.obsm:
        raise ValueError(f"Coordinates '{coordinates_key}' not found in adata.obsm")
    
    coords = adata.obsm[coordinates_key]
    
    # Get clustering columns for this key
    clustering_cols = [col for col in adata.obs.columns if col.startswith(f'{key}_')]
    
    if not clustering_cols:
        raise ValueError(f"No multiresolution clustering results found for key '{key}'")
    
    # Extract resolution values from column names
    available_resolutions = []
    clustering_data = {}
    
    for col in clustering_cols:
        try:
            res = float(col.split('_')[-1])
            available_resolutions.append(res)
            clustering_data[res] = adata.obs[col].astype(int).values
        except ValueError:
            continue
    
    # Filter to requested resolutions if provided
    if resolutions is not None:
        available_resolutions = [r for r in available_resolutions if r in resolutions]
        clustering_data = {r: clustering_data[r] for r in available_resolutions if r in clustering_data}
    
    if not clustering_data:
        raise ValueError("No valid clustering data found")
    
    # Sort by resolution
    sorted_resolutions = sorted(clustering_data.keys())
    
    # Prepare data for plot_multiresolution_graph
    clusterings = [clustering_data[r] for r in sorted_resolutions]
    
    # Get confidence data if available
    confidence = None
    if confidence_key and confidence_key in adata.obs:
        confidence = adata.obs[confidence_key].values
    elif f'{key}_confidence' in adata.obs:
        confidence = adata.obs[f'{key}_confidence'].values
    
    # Extract adjacency matrix for graph overlay (same logic as graph function)
    adjacency = None
    if show_graph:
        # First try standard scanpy location in adata.obsp
        if adjacency_key in adata.obsp:
            adjacency = adata.obsp[adjacency_key]
        # Then try treeclust-specific keys in adata.obsp
        elif f'{neighbors_key}_{adjacency_key}' in adata.obsp:
            adjacency = adata.obsp[f'{neighbors_key}_{adjacency_key}']
        # Finally try the old location in adata.uns[neighbors_key]
        elif neighbors_key in adata.uns:
            neighbors_info = adata.uns[neighbors_key]
            if adjacency_key in neighbors_info:
                adjacency = neighbors_info[adjacency_key]
    
    # Use empty matrix if no adjacency found or show_graph is False
    if adjacency is None:
        adjacency = sparse.csr_matrix((coords.shape[0], coords.shape[0]))
    
    # Create subplot grid for different resolutions
    if skip_repeated:
        n_clusters_prev = -1
        n_resolutions = 0
        for res in sorted_resolutions:
            n_clusters = len(np.unique(clustering_data[res]))
            if n_clusters > n_clusters_prev:
                n_resolutions += 1
                n_clusters_prev = n_clusters
    else:
        n_resolutions = len(sorted_resolutions)
    n_cols = min(n_cols, n_resolutions)
    n_rows = int(np.ceil(n_resolutions / n_cols))
    
    if figsize == None:
        figsize = (5*n_cols, 4*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_resolutions == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes) if n_resolutions > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each resolution using plot_graph
    count = 0
    n_clusters_prev = 0
    for i, res in enumerate(sorted_resolutions):
        if count >= len(axes):
            break
            
        res_key = f"{key}_{i}"
        if res_key in adata.obs:
            # Create individual plot for this resolution
            ax = axes[count]
            
            # Use plot_graph for individual subplot
            color_data = adata.obs[res_key].values

            n_clusters = len(color_data.unique())
            if n_clusters_prev < n_clusters or not skip_repeated:
                count += 1
                n_clusters_prev = n_clusters
            else:
                continue
            
            plot_graph(
                edge_width=adjacency,  # Use actual adjacency matrix or empty matrix
                node_pos=coords,
                node_color=color_data,
                show_labels=show_cluster_labels,
                ax=ax,
                **kwargs
            )
            
            # Set title manually after plotting
            ax.set_title(f'Resolution {i}')
            ax.axis('off')
            ax.grid(False)
            ax.legend(loc=(1.01, 0.0))
    
    # Hide empty subplots
    for i in range(count, len(axes)):
        axes[i].set_visible(False)
        
    return fig

def clustering_comparison(
    adata: 'anndata.AnnData',
    *,
    keys: Sequence[str],
    coordinates_key: str = 'X_umap',
    palette: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Compare multiple clustering results side by side.
    
    Plots different clustering results in a grid layout for easy comparison
    of how different methods or parameters affect cluster assignment.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    keys : sequence of str
        Keys in adata.obs containing clustering results to compare.
        
    coordinates_key : str, default='X_umap'
        Key in adata.obsm for coordinates to plot.
        
    palette : str, optional
        Color palette for clusters.
        
    save : str, optional
        Save figure to file.
        
    **kwargs
        Additional arguments passed to plot_clustering_comparison.
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object if save is None.
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> tcsc.pl.clustering_comparison(
    ...     adata, 
    ...     keys=['leiden', 'louvain', 'treeclust_leiden']
    ... )
    """
    if not HAS_ANNDATA:
        raise ImportError("AnnData is required for plotting functions")
    
    if coordinates_key not in adata.obsm:
        raise ValueError(f"Coordinates '{coordinates_key}' not found in adata.obsm")
    
    coords = adata.obsm[coordinates_key]
    
    # Extract clustering data
    clusterings = []
    method_names = []
    
    for key in keys:
        if key in adata.obs:
            clusterings.append(adata.obs[key].astype(int).values)
            method_names.append(key)
        else:
            warnings.warn(f"Key '{key}' not found in adata.obs, skipping")
    
    if not clusterings:
        raise ValueError("No valid clustering data found")
    
    # Use treeclust's comparison plotting
    fig = plot_clustering_comparison(
        coordinates=coords,
        clusterings=clusterings,
        method_names=method_names,
        palette=palette,
        save=save,
        **kwargs
    )
    
    return fig

def multiresolution_hierarchy(
    adata: 'anndata.AnnData',
    *,
    key: str = 'multiresolution_leiden',
    save: Optional[str] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Plot hierarchical structure from multiresolution clustering.
    
    Visualizes how clusters split and merge across different resolution
    levels, showing the hierarchical relationship between clusters.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    key : str, default='multiresolution_leiden'
        Key in adata.uns containing multiresolution results.
        
    save : str, optional
        Save figure to file.
        
    **kwargs
        Additional arguments passed to plot_multiresolution_hierarchy.
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object if save is None.
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> tcsc.pl.multiresolution_hierarchy(adata, key='multires_leiden')
    """
    if not HAS_ANNDATA:
        raise ImportError("AnnData is required for plotting functions")
    
    if key not in adata.uns:
        raise ValueError(f"Key '{key}' not found in adata.uns")
        
    multires_info = adata.uns[key]

    if 'hierarchical' not in multires_info or not multires_info['hierarchical']:
        raise ValueError(f"No hierarchy information found in '{key}'")

    resolutions = multires_info.get('resolutions', [])
    labels = {i: adata.obs[f'{key}_{r}'].values for i, r in enumerate(resolutions)}
    
    # Use treeclust's hierarchy plotting
    fig = plot_multiresolution_hierarchy(
        labels,
        **kwargs
    )
    
    return fig

def dotplot(
    adata: 'anndata.AnnData',
    groupby: Union[str, Sequence[str]],
    gene_symbols: Optional[str] = None,
    use_raw: bool = True,
    group_rows: bool = True,
    group_cols: bool = True,
    normalize_cols: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Enhanced dotplot showing gene expression across multiple grouping variables.
    
    Creates a comprehensive dotplot visualization where dot size represents the 
    proportion of cells expressing each gene, and dot color represents the mean
    expression level. Supports multiple grouping variables with hierarchical
    clustering and visual separation.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression data and cell annotations.
        
    groupby : str or sequence of str
        Key(s) in adata.obs for grouping cells (e.g., 'leiden', 'cell_type').
        Multiple grouping variables will be displayed with visual separation.
        
    gene_symbols : str, optional
        List of gene symbols to display. If None, uses all genes.
        
    use_raw : bool, default=True
        Whether to use raw expression data (adata.raw.X) if available.
        If False or raw is None, uses processed data (adata.X).
        
    group_rows : bool, default=True
        Whether to perform hierarchical clustering on rows (cell groups).
        Clusters similar expression patterns together within each groupby category.
        
    group_cols : bool, default=True
        Whether to perform hierarchical clustering on columns (genes).
        Clusters genes with similar expression patterns across all groups.
        
    normalize_cols : bool, default=True
        Whether to normalize expression values for better visualization.
        
    ax : matplotlib.Axes, optional
        Pre-existing axes for the plot. If None, creates a new figure.
        
    **kwargs
        Additional keyword arguments (currently unused).
        
    Returns
    -------
    fig : matplotlib.Figure or None
        Figure object containing the plot. Returns None if ax is provided.
        
    Notes
    -----
    The dotplot visualization encodes two types of information:
    
    - **Dot size**: Proportion of cells in each group that express the gene
      (non-zero expression values). Larger dots indicate more cells expressing.
      
    - **Dot color**: Mean expression level of the gene in each group.
      Color intensity represents expression magnitude.
      
    Multiple grouping variables are displayed with visual separation between
    categories. Each group is labeled as 'groupby_cluster' (e.g., 'leiden_0').
    
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> # Basic dotplot with single grouping
    >>> tcsc.pl.dotplot(adata, groupby='leiden', 
    ...                 gene_symbols=['CD4', 'CD8A', 'CD19'])
    >>> 
    >>> # Multiple grouping variables
    >>> tcsc.pl.dotplot(adata, groupby=['leiden', 'cell_type'],
    ...                 gene_symbols=['CD4', 'CD8A', 'CD19'])
    >>> 
    >>> # Without clustering
    >>> tcsc.pl.dotplot(adata, groupby='leiden',
    ...                 gene_symbols=['CD4', 'CD8A'],
    ...                 group_rows=False, group_cols=False)
    """
    
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram

    if type(groupby) is str:
        groupby = [groupby]

    gs = []
    for group in groupby:

        if group not in adata.obs:
            raise ValueError(f"Group '{group}' not found in adata.obs. Available groups: {list(adata.obs.columns)}")

        labels = adata.obs[group].values
        group_adata = adata[:,gene_symbols]
        if use_raw and adata.raw is not None:
            X = group_adata.raw[:,gene_symbols].X
        elif use_raw and adata.raw is None:
            warnings.warn("adata.raw is None, cannot use use_raw=True")
            X = group_adata.X
        else:
            X = group_adata.X

        if sparse.issparse(X):
            X = X.toarray()

        df = pd.DataFrame(X, index=labels, columns=gene_symbols)
        g_mean = df.groupby(labels).mean()
        
        # Calculate proportion of non-zero values for each group
        g_nonzero_prop = df.groupby(labels).apply(lambda x: (x > 0).mean())
        
        if group_rows:            
            # Calculate pairwise distances between groups (use mean for clustering)
            distances = pdist(g_mean.values, metric='euclidean')
            # Perform hierarchical clustering
            linkage_matrix = linkage(distances, method='average')
            # Get the order from clustering
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            # Reorder both DataFrames
            g_mean = g_mean.iloc[cluster_order]
            g_nonzero_prop = g_nonzero_prop.iloc[cluster_order]
        else:
            g_mean = g_mean.sort_index()
            g_nonzero_prop = g_nonzero_prop.sort_index()

        # Add multiindex with group information
        g_mean.index = pd.MultiIndex.from_product([[group], g_mean.index], names=['groupby', 'cluster'])
        g_nonzero_prop.index = pd.MultiIndex.from_product([[group], g_nonzero_prop.index], names=['groupby', 'cluster'])
        
        gs.append((g_mean, g_nonzero_prop))
    
    # Stack all group results by rows
    combined_g_mean = pd.concat([g[0] for g in gs], axis=0)
    if normalize_cols:
        combined_g_mean = (combined_g_mean - combined_g_mean.min()) / (combined_g_mean.max() - combined_g_mean.min())
    combined_g_nonzero = pd.concat([g[1] for g in gs], axis=0)

    if group_cols:
        # Cluster columns (genes) using mean expression
        distances = pdist(combined_g_mean.values.T, metric='euclidean')
        linkage_matrix = linkage(distances, method='average')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        gene_order = dendro['leaves']
        combined_g_mean = combined_g_mean.iloc[:, gene_order]
        combined_g_nonzero = combined_g_nonzero.iloc[:, gene_order]
    else:
        combined_g_mean = combined_g_mean.sort_index(axis=1)
        combined_g_nonzero = combined_g_nonzero.sort_index(axis=1)

    # Create scatterplot visualization
    if ax is None:
        n_genes = len(combined_g_mean.columns)
        n_groups = len(combined_g_mean.index.get_level_values(0).unique())
        fig, ax = plt.subplots(figsize=(n_genes * 0.8, len(combined_g_mean) * 0.4 + n_groups * 0.5))
    else:
        fig = None
    
    # Prepare data for scatterplot
    y_positions = []
    y_labels = []
    y_tick_positions = []
    x_positions = []
    sizes = []
    colors = []
    
    # Add separation between different groupby categories
    current_y = 0
    groupby_separations = {}
    
    for groupby_name in combined_g_mean.index.get_level_values(0).unique():
        groupby_separations[groupby_name] = current_y
        group_data_mean = combined_g_mean.loc[groupby_name]
        group_data_nonzero = combined_g_nonzero.loc[groupby_name]
        
        for i, (cluster_name, row_mean) in enumerate(group_data_mean.iterrows()):
            row_nonzero = group_data_nonzero.loc[cluster_name]
            y_pos = current_y + i
            y_positions.extend([y_pos] * len(combined_g_mean.columns))
            y_labels.append(f"{groupby_name}_{cluster_name}")
            y_tick_positions.append(y_pos)
            
            for j, gene in enumerate(combined_g_mean.columns):
                x_positions.append(j)
                expr_value = row_mean[gene]
                nonzero_prop = row_nonzero[gene]
                
                # Size based on proportion of non-zero values
                sizes.append(20 + nonzero_prop * 180)  # Scale from 20 to 200
                colors.append(expr_value)
        
        # Add separation for next group
        current_y += len(group_data_mean) + 1  # +1 for separation
    
    # Create the scatter plot
    scatter = ax.scatter(x_positions, y_positions, s=sizes, c=colors, **kwargs)
    
    # Customize plot
    ax.set_xticks(range(len(combined_g_mean.columns)))
    ax.set_xticklabels(combined_g_mean.columns, rotation=45, ha='right')
    ax.set_xlabel('Genes')
    
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Groups')
        
    # Create color legend elements (expression levels)
    expr_min, expr_max = min(colors), max(colors)
    color_levels = [expr_min, (expr_min + expr_max) / 2, expr_max]
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=expr_min, vmax=expr_max)
    
    color_legend_elements = []
    for level in color_levels:
        color_legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=cmap(norm(level)), 
                   markersize=8, 
                   label=f'{level:.2f}')
        )
    
    # Create size legend elements (proportion of expressing cells)
    size_levels = [0.0, 0.5, 1.0]
    size_legend_elements = []
    for prop in size_levels:
        size = (20 + prop * 180) / 10  # Scale down for legend
        size_legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor='gray', 
                   markersize=size**0.5,  # Square root for visual scaling
                   label=f'{prop:.0%}')
        )
    
    # Add legends
    color_legend = ax.legend(handles=color_legend_elements, 
                            title='Mean Expression', 
                            loc='upper left', 
                            bbox_to_anchor=(1.02, 1))
    
    size_legend = ax.legend(handles=size_legend_elements, 
                           title='% Expressing', 
                           loc='upper left', 
                           bbox_to_anchor=(1.02, 0.7))
    
    # Add the color legend back (matplotlib removes previous legend when adding new one)
    ax.add_artist(color_legend)
    plt.tight_layout()
    
    return fig
