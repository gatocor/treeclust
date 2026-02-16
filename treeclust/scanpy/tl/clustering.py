"""
Scanpy-style clustering functions for treeclust.

This module provides scanpy-compatible clustering functions that compute
clusterings using treeclust methods and store results in AnnData objects
following scanpy conventions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, Sequence
import warnings

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
    warnings.warn("AnnData not available. Functions will work with arrays but not AnnData objects.")

from ...clustering import (
    Leiden,
    Louvain,
    MultiresolutionLeiden,
    MultiresolutionLouvain
)

def leiden(
    adata: Union['anndata.AnnData', np.ndarray],
    resolution: float = 0.5,
    *,
    neighbors_key: Optional[str] = 'neighbors',
    adjacency: Optional[np.ndarray] = None,
    use_weights: bool = True,
    random_state: Optional[int] = 0,
    partition_type: str = 'RB',
    n_repetitions: int = 1,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Perform Leiden clustering on neighborhood graph using treeclust.
    
    Similar to scanpy.tl.leiden but uses treeclust's robust Leiden implementation
    with additional options for consensus clustering and GPU acceleration.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or adjacency matrix.
        
    resolution : float, default=0.5
        Resolution parameter for modularity optimization.
        Higher values lead to more clusters.
        
    neighbors_key : str, optional, default='neighbors'
        Key in adata.uns containing neighborhood information.
        
    adjacency : np.ndarray, optional
        Custom adjacency matrix to use instead of neighbors.
        
    use_weights : bool, default=True
        Whether to use edge weights from the adjacency matrix.
        
    n_iterations : int, default=-1
        Number of iterations. -1 means run until convergence.
        
    random_state : int, optional
        Random state for reproducibility.
        
    key_added : str, optional
        Key under which to add the clustering results.
        If None, uses 'leiden'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to treeclust.Leiden.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
        Adds clustering results to adata.obs[key_added].
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> tcsc.tl.leiden(adata, resolution=0.5)
    >>> print(adata.obs['leiden'])
    """
    if key_added is None:
        key_added = 'leiden'
        
    adata = adata.copy() if copy else adata
    
    # Extract adjacency matrix
    if adjacency is not None:
        adj_matrix = adjacency
    elif HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adj_matrix = None
        
        # First try to get from adata.uns[neighbors_key] (treeclust style)
        if neighbors_key in adata.uns:
            neighbors_info = adata.uns[neighbors_key]
            if 'connectivities' in neighbors_info:
                adj_matrix = neighbors_info['connectivities']
            elif 'distances' in neighbors_info:
                adj_matrix = neighbors_info['distances']
        
        # If not found, try standard scanpy locations in adata.obsp
        if adj_matrix is None:
            if neighbors_key == 'neighbors':
                # Standard scanpy case - look in obsp
                if 'connectivities' in adata.obsp:
                    adj_matrix = adata.obsp['connectivities']
                elif 'distances' in adata.obsp:
                    adj_matrix = adata.obsp['distances']
            else:
                # Custom neighbors_key - try with suffix
                connectivities_key = f"{neighbors_key}_connectivities"
                distances_key = f"{neighbors_key}_distances"
                if connectivities_key in adata.obsp:
                    adj_matrix = adata.obsp[connectivities_key]
                elif distances_key in adata.obsp:
                    adj_matrix = adata.obsp[distances_key]
        
        if adj_matrix is None:
            raise ValueError(f"No adjacency information found for '{neighbors_key}'. "
                           f"Available adata.uns keys: {list(adata.uns.keys())}, "
                           f"Available adata.obsp keys: {list(adata.obsp.keys())}")
    else:
        adj_matrix = adata
    
    # Perform Leiden clustering
    leiden_clusterer = Leiden(
        resolution=resolution,
        random_state=random_state,
        partition_type=partition_type,
        n_repetitions=n_repetitions,
        **kwargs
    )
    
    # Keep the matrix in its original format (sparse is preferred)
    cluster_labels = leiden_clusterer.fit_predict(adj_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adata.obs[key_added] = pd.Categorical(np.array(cluster_labels).astype(str))
        
        # Store clustering parameters
        adata.uns[f'{key_added}_params'] = {
            'resolution': resolution,
            'partition_type': partition_type,
            'n_repetitions': n_repetitions,
            'use_weights': use_weights,
            'method': 'leiden',
            'random_state': random_state
        }
    else:
        return cluster_labels
    
    return adata if copy else None


def louvain(
    adata: Union['anndata.AnnData', np.ndarray],
    resolution: float = 0.5,
    *,
    neighbors_key: Optional[str] = 'neighbors',
    adjacency: Optional[np.ndarray] = None,
    use_weights: bool = True,
    random_state: Optional[int] = 0,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Perform Louvain clustering on neighborhood graph using treeclust.
    
    Similar to scanpy.tl.louvain but uses treeclust's Louvain implementation
    with additional robustness features.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or adjacency matrix.
        
    resolution : float, default=0.5
        Resolution parameter for modularity optimization.
        
    neighbors_key : str, optional, default='neighbors'
        Key in adata.uns containing neighborhood information.
        
    adjacency : np.ndarray, optional
        Custom adjacency matrix to use instead of neighbors.
        
    use_weights : bool, default=True
        Whether to use edge weights from the adjacency matrix.
        
    random_state : int, optional
        Random state for reproducibility.
        
    key_added : str, optional
        Key under which to add the clustering results.
        If None, uses 'louvain'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to treeclust.Louvain.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
        Adds clustering results to adata.obs[key_added].
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> tcsc.tl.louvain(adata, resolution=0.8)
    >>> print(adata.obs['louvain'])
    """
    if key_added is None:
        key_added = 'louvain'
        
    adata = adata.copy() if copy else adata
    
    # Extract adjacency matrix
    if adjacency is not None:
        adj_matrix = adjacency
    elif HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adj_matrix = None
        
        # First try to get from adata.uns[neighbors_key] (treeclust style)
        if neighbors_key in adata.uns:
            neighbors_info = adata.uns[neighbors_key]
            if 'connectivities' in neighbors_info:
                adj_matrix = neighbors_info['connectivities']
            elif 'distances' in neighbors_info:
                adj_matrix = neighbors_info['distances']
        
        # If not found, try standard scanpy locations in adata.obsp
        if adj_matrix is None:
            if neighbors_key == 'neighbors':
                # Standard scanpy case - look in obsp
                if 'connectivities' in adata.obsp:
                    adj_matrix = adata.obsp['connectivities']
                elif 'distances' in adata.obsp:
                    adj_matrix = adata.obsp['distances']
            else:
                # Custom neighbors_key - try with suffix
                connectivities_key = f"{neighbors_key}_connectivities"
                distances_key = f"{neighbors_key}_distances"
                if connectivities_key in adata.obsp:
                    adj_matrix = adata.obsp[connectivities_key]
                elif distances_key in adata.obsp:
                    adj_matrix = adata.obsp[distances_key]
        
        if adj_matrix is None:
            raise ValueError(f"No adjacency information found for '{neighbors_key}'. "
                           f"Available adata.uns keys: {list(adata.uns.keys())}, "
                           f"Available adata.obsp keys: {list(adata.obsp.keys())}")
    else:
        adj_matrix = adata
    
    # Perform Louvain clustering
    louvain_clusterer = Louvain(
        resolution=resolution,
        random_state=random_state,
        **kwargs
    )
    
    # Keep the matrix in its original format (sparse is preferred)
    cluster_labels = louvain_clusterer.fit_predict(adj_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adata.obs[key_added] = pd.Categorical(np.array(cluster_labels).astype(str))
        
        # Store clustering parameters
        adata.uns[f'{key_added}_params'] = {
            'resolution': resolution,
            'use_weights': use_weights,
            'method': 'louvain',
            'random_state': random_state
        }
    else:
        return cluster_labels
    
    return adata if copy else None

def multiresolution_leiden(
    adata: Union['anndata.AnnData', np.ndarray],
    resolutions: Union[Sequence[float], tuple] = (0, 1),
    *,
    neighbors_key: Optional[str] = 'neighbors',
    adjacency: Optional[np.ndarray] = None,
    use_weights: bool = True,
    random_state: Optional[int] = 0,
    partition_type: str = 'RB',
    n_repetitions: int = 10,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Perform multiresolution Leiden clustering using treeclust.
    
    Performs Leiden clustering at multiple resolution levels and stores
    results as a hierarchical clustering tree. Provides consensus
    clustering across resolutions for robust cluster identification.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or adjacency matrix.
        
    resolutions : sequence of float, default=(0.1, 0.3, 0.5, 0.7, 1.0)
        Resolution parameters to test.
        
    neighbors_key : str, optional, default='neighbors'
        Key in adata.uns containing neighborhood information.
        
    adjacency : np.ndarray, optional
        Custom adjacency matrix to use instead of neighbors.
        
    use_weights : bool, default=True
        Whether to use edge weights from the adjacency matrix.
        
    n_iterations : int, default=-1
        Number of iterations. -1 means run until convergence.
        
    random_state : int, optional
        Random state for reproducibility.
        
    key_added : str, optional
        Key under which to add the clustering results.
        If None, uses 'multiresolution_leiden'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to treeclust.MultiresolutionLeiden.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
        Adds clustering results for each resolution to adata.obs
        and hierarchical information to adata.uns.
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> # Use resolution range (automatic selection between 0.1 and 2.0)
    >>> tcsc.tl.multiresolution_leiden(adata, resolutions=(0.1, 2.0))
    >>> # Use specific resolution values
    >>> tcsc.tl.multiresolution_leiden(adata, resolutions=[0.2, 0.5, 1.0])
    >>> # Use default range (0 to 1)
    >>> tcsc.tl.multiresolution_leiden(adata)
    >>> print(adata.obs['multiresolution_leiden_0.5'])
    >>> print(adata.uns['multiresolution_leiden']['hierarchy'])
    """
    if key_added is None:
        key_added = 'multiresolution_leiden'
    
    # Handle resolutions parameter: tuple = range, list/sequence = specific values
    if isinstance(resolutions, tuple) and len(resolutions) == 2:
        # Tuple interpreted as (min_resolution, max_resolution) range
        resolution_range = resolutions
        resolution_values = None
    else:
        # List/sequence interpreted as specific resolution values
        resolution_values = list(resolutions)
        resolution_range = None
        
    adata = adata.copy() if copy else adata
    
    # Extract adjacency matrix
    if adjacency is not None:
        adj_matrix = adjacency
    elif HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adj_matrix = None
        
        # First try to get from adata.uns[neighbors_key] (treeclust style)
        if neighbors_key in adata.uns:
            neighbors_info = adata.uns[neighbors_key]
            if 'connectivities' in neighbors_info:
                adj_matrix = neighbors_info['connectivities']
            elif 'distances' in neighbors_info:
                adj_matrix = neighbors_info['distances']
        
        # If not found, try standard scanpy locations in adata.obsp
        if adj_matrix is None:
            if neighbors_key == 'neighbors':
                # Standard scanpy case - look in obsp
                if 'connectivities' in adata.obsp:
                    adj_matrix = adata.obsp['connectivities']
                elif 'distances' in adata.obsp:
                    adj_matrix = adata.obsp['distances']
            else:
                # Custom neighbors_key - try with suffix
                connectivities_key = f"{neighbors_key}_connectivities"
                distances_key = f"{neighbors_key}_distances"
                if connectivities_key in adata.obsp:
                    adj_matrix = adata.obsp[connectivities_key]
                elif distances_key in adata.obsp:
                    adj_matrix = adata.obsp[distances_key]
        
        if adj_matrix is None:
            raise ValueError(f"No adjacency information found for '{neighbors_key}'. "
                           f"Available adata.uns keys: {list(adata.uns.keys())}, "
                           f"Available adata.obsp keys: {list(adata.obsp.keys())}")
    else:
        adj_matrix = adata
    
    # Perform multiresolution Leiden clustering
    multiresolution_clusterer = MultiresolutionLeiden(
        resolution_values=resolution_values,
        resolution_range=resolution_range,
        random_state=random_state,
        partition_type=partition_type,
        n_repetitions=n_repetitions,
        **kwargs
    )
    
    # Keep the matrix in its original format (sparse is preferred)
    clustering_results = multiresolution_clusterer.fit_predict(adj_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, ad.AnnData):
        # Get actual resolutions used (either from range or specific values)
        if resolution_range is not None:
            # For range mode, get actual resolutions from the fitted model
            actual_resolutions = multiresolution_clusterer.resolution_values_
        else:
            # For specific values mode, use what was provided
            actual_resolutions = resolution_values
        
        # Store clustering for each resolution
        for i, res in enumerate(actual_resolutions):
            res_key = f'{key_added}_{res}'
            if i in clustering_results:
                cluster_labels = clustering_results[i]
            else:
                # Try to find closest key (sometimes floating point precision issues)
                available_keys = list(clustering_results.keys())
                closest_key = min(available_keys, key=lambda x: abs(x - res) if isinstance(x, (int, float)) else float('inf'))
                cluster_labels = clustering_results[closest_key]
            adata.obs[res_key] = pd.Categorical(np.array(cluster_labels).astype(str))
        
        # Store hierarchical information and parameters
        adata.uns[key_added] = {
            'resolutions': list(actual_resolutions),
            'resolution_range': resolution_range if resolution_range is not None else None,
            'resolution_mode': 'range' if resolution_range is not None else 'values',
            'hierarchy': multiresolution_clusterer.hierarchy_ if hasattr(multiresolution_clusterer, 'hierarchy_') else None,
            'confidence': multiresolution_clusterer.confidence_ if hasattr(multiresolution_clusterer, 'confidence_') else None,
            'method': 'multiresolution_leiden',
            'hierarchical': True
        }
        
        adata.uns[f'{key_added}_params'] = {
            'resolutions': list(actual_resolutions),
            'resolution_range': resolution_range,
            'resolution_mode': 'range' if resolution_range is not None else 'values',
            'partition_type': partition_type,
            'n_repetitions': n_repetitions,
            'use_weights': use_weights,
            'method': 'multiresolution_leiden',
            'random_state': random_state
        }
    else:
        return clustering_results
    
    return adata if copy else None


def multiresolution_louvain(
    adata: Union['anndata.AnnData', np.ndarray],
    resolutions: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 1.0),
    *,
    neighbors_key: Optional[str] = 'neighbors',
    adjacency: Optional[np.ndarray] = None,
    use_weights: bool = True,
    random_state: Optional[int] = 0,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Perform multiresolution Louvain clustering using treeclust.
    
    Performs Louvain clustering at multiple resolution levels and stores
    results as a hierarchical clustering tree.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or adjacency matrix.
        
    resolutions : sequence of float, default=(0.1, 0.3, 0.5, 0.7, 1.0)
        Resolution parameters to test.
        
    neighbors_key : str, optional, default='neighbors'
        Key in adata.uns containing neighborhood information.
        
    adjacency : np.ndarray, optional
        Custom adjacency matrix to use instead of neighbors.
        
    use_weights : bool, default=True
        Whether to use edge weights from the adjacency matrix.
        
    random_state : int, optional
        Random state for reproducibility.
        
    key_added : str, optional
        Key under which to add the clustering results.
        If None, uses 'multiresolution_louvain'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to treeclust.MultiresolutionLouvain.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
        Adds clustering results for each resolution to adata.obs
        and hierarchical information to adata.uns.
        
    Examples
    --------
    >>> import treeclust.scanpy as tcsc
    >>> tcsc.tl.multiresolution_louvain(adata, resolutions=(0.2, 0.5, 1.0))
    >>> print(adata.obs['multiresolution_louvain_0.5'])
    >>> print(adata.uns['multiresolution_louvain']['hierarchy'])
    """
    if key_added is None:
        key_added = 'multiresolution_louvain'
        
    adata = adata.copy() if copy else adata
    
    # Extract adjacency matrix
    if adjacency is not None:
        adj_matrix = adjacency
    elif HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adj_matrix = None
        
        # First try to get from adata.uns[neighbors_key] (treeclust style)
        if neighbors_key in adata.uns:
            neighbors_info = adata.uns[neighbors_key]
            if 'connectivities' in neighbors_info:
                adj_matrix = neighbors_info['connectivities']
            elif 'distances' in neighbors_info:
                adj_matrix = neighbors_info['distances']
        
        # If not found, try standard scanpy locations in adata.obsp
        if adj_matrix is None:
            if neighbors_key == 'neighbors':
                # Standard scanpy case - look in obsp
                if 'connectivities' in adata.obsp:
                    adj_matrix = adata.obsp['connectivities']
                elif 'distances' in adata.obsp:
                    adj_matrix = adata.obsp['distances']
            else:
                # Custom neighbors_key - try with suffix
                connectivities_key = f"{neighbors_key}_connectivities"
                distances_key = f"{neighbors_key}_distances"
                if connectivities_key in adata.obsp:
                    adj_matrix = adata.obsp[connectivities_key]
                elif distances_key in adata.obsp:
                    adj_matrix = adata.obsp[distances_key]
        
        if adj_matrix is None:
            raise ValueError(f"No adjacency information found for '{neighbors_key}'. "
                           f"Available adata.uns keys: {list(adata.uns.keys())}, "
                           f"Available adata.obsp keys: {list(adata.obsp.keys())}")
    else:
        adj_matrix = adata
    
    # Perform multiresolution Louvain clustering
    multiresolution_clusterer = MultiresolutionLouvain(
        resolution_values=resolutions,
        random_state=random_state,
        **kwargs
    )
    
    # Keep the matrix in its original format (sparse is preferred)
    clustering_results = multiresolution_clusterer.fit_predict(adj_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, ad.AnnData):
        # Store clustering for each resolution
        for i, res in enumerate(resolutions):
            res_key = f'{key_added}_{res}'
            if i in clustering_results:
                cluster_labels = clustering_results[i]
            else:
                # Fallback to direct resolution key
                cluster_labels = clustering_results.get(res, clustering_results[list(clustering_results.keys())[i]])
            adata.obs[res_key] = pd.Categorical(np.array(cluster_labels).astype(str))
        
        # Store hierarchical information and parameters
        adata.uns[key_added] = {
            'resolutions': list(resolutions),
            'hierarchy': multiresolution_clusterer.hierarchy_ if hasattr(multiresolution_clusterer, 'hierarchy_') else None,
            'confidence': multiresolution_clusterer.confidence_ if hasattr(multiresolution_clusterer, 'confidence_') else None,
            'method': 'multiresolution_louvain',
            'hierarchical': True
        }
        
        adata.uns[f'{key_added}_params'] = {
            'resolutions': list(resolutions),
            'use_weights': use_weights,
            'method': 'multiresolution_louvain',
            'random_state': random_state
        }
    else:
        return clustering_results
    
    return adata if copy else None