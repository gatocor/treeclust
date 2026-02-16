"""
Scanpy-style preprocessing functions for treeclust neighbors methods.

This module provides scanpy-compatible preprocessing functions that compute
neighborhoods using various treeclust methods and store results in AnnData objects.
All methods use treeclust's original connectivity computation to preserve sparsity
and intended behavior.
"""

from typing import Optional, Union, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import anndata

import numpy as np
from scipy import sparse
import warnings

try:
    import anndata
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
    anndata = None
    warnings.warn("AnnData not available. Functions will work with arrays but not AnnData objects.")

from ...neighbors import (
    NearestNeighbors,
    ConsensusNearestNeighbors,
    MutualNearestNeighbors,
    ConsensusMutualNearestNeighbors,
    CoassociationDistanceMatrix
)
from ...neighbors.utils import to_connectivities_matrix, to_transition_matrix

def neighbors(
    adata: Union['anndata.AnnData', np.ndarray],
    n_neighbors: int = 15,
    method: str = 'knn',
    metric: str = 'euclidean',
    use_rep: Optional[str] = None,
    use_highly_variable: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Compute neighborhood graph using standard k-nearest neighbors.
    
    This is equivalent to scanpy's pp.neighbors but uses treeclust's
    NearestNeighbors implementation with original connectivity computation.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or data matrix.
        
    n_neighbors : int, default=15
        Number of nearest neighbors to compute.
        
    method : str, default='knn'
        Method for neighborhood computation. Currently only 'knn' supported.
        
    metric : str, default='euclidean'
        Distance metric to use for neighborhood computation.
        
    use_rep : str, optional
        Use the indicated representation. If None, uses .X.
        
    use_highly_variable : bool, default=True
        Whether to use highly variable genes only. If True and use_rep is None,
        uses adata.X[:, adata.var.highly_variable] if available.
        
    key_added : str, optional
        Key under which to add the computed distances and connectivities.
        If None, uses 'neighbors'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to NearestNeighbors.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
        Adds the following to adata.uns:
        - 'neighbors': dict with 'connectivities', 'distances', 'transitions'
        - 'neighbors_params': parameters used
    """
    if key_added is None:
        key_added = 'neighbors'
    
    adata = adata.copy() if copy else adata
    
    # Extract data matrix (following scanpy's use_rep logic)
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        if use_rep is not None:
            if use_rep == 'X':
                X = adata.X
            else:
                X = adata.obsm[use_rep]
        else:
            # Scanpy's default behavior: use X if n_vars < 50, otherwise X_pca
            if adata.n_vars < 50:
                X = adata.X
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                X = adata.obsm['X_pca']
            else:
                # Fall back to X if no PCA available
                X = adata.X
        
        # Apply highly variable genes filter if requested and applicable
        if use_highly_variable and use_rep is None and hasattr(adata, 'var') and 'highly_variable' in adata.var.columns:
            if sparse.issparse(X):
                X = X[:, adata.var.highly_variable.values].toarray()
            else:
                X = X[:, adata.var.highly_variable.values]
        elif sparse.issparse(X):
            X = X.toarray()
    else:
        X = adata
    
    # Compute neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Convert distances and indices to a distance matrix efficiently
    # Build lists for COO matrix format
    row_idx = []
    col_idx = []
    data = []
    
    for i in range(X.shape[0]):
        for j, neighbor_idx in enumerate(indices[i]):
            if neighbor_idx != i:  # Skip self-connections
                row_idx.append(i)
                col_idx.append(neighbor_idx)
                data.append(distances[i][j])
    
    # Create sparse matrix efficiently using COO format, then convert to CSR
    distance_matrix = sparse.coo_matrix(
        (data, (row_idx, col_idx)), 
        shape=(X.shape[0], X.shape[0])
    ).tocsr()
    
    # Convert to connectivities using treeclust's original implementation
    connectivities = to_connectivities_matrix(distance_matrix, k_neighbors=n_neighbors)
    transitions = to_transition_matrix(distance_matrix)
    
    # Store results (scanpy-compatible format)
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        # Store matrices in obsp (scanpy convention)
        connectivities_key = f'{key_added}_connectivities' if key_added != 'neighbors' else 'connectivities'
        distances_key = f'{key_added}_distances' if key_added != 'neighbors' else 'distances'
        
        adata.obsp[connectivities_key] = connectivities
        adata.obsp[distances_key] = distance_matrix
        
        # Store metadata in uns (scanpy convention)
        adata.uns[key_added] = {
            'connectivities_key': connectivities_key,
            'distances_key': distances_key,
            'params': {
                'n_neighbors': n_neighbors,
                'method': 'umap',  # Set to 'umap' for scanpy compatibility
                'metric': metric,
                'random_state': 0  # Add for full scanpy compatibility
            }
        }
        
        # Also store matrices directly for backward compatibility with treeclust
        adata.uns[key_added].update({
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        })
    else:
        return {
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    
    return adata if copy else None

def consensus_neighbors(
    adata: Union['anndata.AnnData', np.ndarray],
    n_neighbors: int = 15,
    n_splits: int = 10,
    metric: str = 'euclidean',
    use_rep: Optional[str] = None,
    use_highly_variable: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
    random_state: Optional[int] = 0,
    pipeline_bootstrapper: Optional[Any] = None,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Compute consensus neighborhood graph using bootstrap sampling.
    
    Creates robust neighborhoods by averaging over multiple bootstrap
    samples, reducing sensitivity to outliers and sampling variations.
    Uses treeclust's original connectivity computation.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or data matrix.
        
    n_neighbors : int, default=15
        Number of nearest neighbors to compute.
        
    n_splits : int, default=10
        Number of bootstrap samples for consensus.
        
    metric : str, default='euclidean'
        Distance metric to use for neighborhood computation.
        
    use_rep : str, optional
        Use the indicated representation. If None, uses .X.
        
    use_highly_variable : bool, default=True
        Whether to use highly variable genes only. If True and use_rep is None,
        uses adata.X[:, adata.var.highly_variable] if available.
        
    key_added : str, optional
        Key under which to add the computed distances and connectivities.
        If None, uses 'consensus_neighbors'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    random_state : int, optional
        Random seed for reproducible results. Controls the randomness of the
        bootstrap sampling and ensures consistent results across runs.
        
    pipeline_bootstrapper : PipelineBootstrapper, optional
        Custom PipelineBootstrapper instance for controlling the bootstrap process.
        If None, creates a default PipelineBootstrapper with 80% sample ratio.
        
    **kwargs
        Additional arguments passed to ConsensusNearestNeighbors.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
    """
    if key_added is None:
        key_added = 'consensus_neighbors'
    
    adata = adata.copy() if copy else adata
    
    # Extract data matrix
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        if use_rep is not None:
            if use_rep == 'X':
                X = adata.X
            else:
                X = adata.obsm[use_rep]
        else:
            if adata.n_vars < 50:
                X = adata.X
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                X = adata.obsm['X_pca']
            else:
                X = adata.X
        
        # Apply highly variable genes filter if requested and applicable
        if use_highly_variable and use_rep == "X" and 'highly_variable' in adata.var.columns:
            if sparse.issparse(X):
                X = X[:, adata.var.highly_variable.values].toarray()
            else:
                X = X[:, adata.var.highly_variable.values]
        elif sparse.issparse(X):
            X = X.toarray()
    else:
        X = adata
    
    # Compute consensus neighbors
    cnn = ConsensusNearestNeighbors(
        n_neighbors=n_neighbors,
        n_splits=n_splits,
        metric=metric,
        random_state=random_state,
        pipeline_bootstrapper=pipeline_bootstrapper,
        **kwargs
    )
    cnn.fit(X)
    
    # Get distance matrix from fitted model
    distance_matrix = cnn.matrix_
    
    # Convert to connectivities using treeclust's original implementation  
    connectivities = to_connectivities_matrix(distance_matrix, k_neighbors=n_neighbors)
    transitions = to_transition_matrix(distance_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        connectivities_key = f'{key_added}_connectivities' if key_added != 'consensus_neighbors' else 'consensus_connectivities'
        distances_key = f'{key_added}_distances' if key_added != 'consensus_neighbors' else 'consensus_distances'
        
        adata.obsp[connectivities_key] = connectivities
        adata.obsp[distances_key] = distance_matrix
        
        adata.uns[key_added] = {
            'connectivities_key': connectivities_key,
            'distances_key': distances_key,
            'params': {
                'n_neighbors': n_neighbors,
                'n_splits': n_splits,
                'method': 'consensus',
                'metric': metric
            },
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    else:
        return {
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    
    return adata if copy else None


def mutual_neighbors(
    adata: Union['anndata.AnnData', np.ndarray],
    n_neighbors: int = 15,
    metric: str = 'euclidean',
    use_rep: Optional[str] = None,
    use_highly_variable: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Compute mutual nearest neighbors graph.
    
    Only includes edges between points that are mutual nearest neighbors,
    creating a sparser, more reliable neighborhood structure.
    Uses treeclust's original connectivity computation to preserve sparsity.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or data matrix.
        
    n_neighbors : int, default=15
        Number of nearest neighbors to compute initially.
        
    metric : str, default='euclidean'
        Distance metric to use for neighborhood computation.
        
    use_rep : str, optional
        Use the indicated representation. If None, uses .X.
        
    use_highly_variable : bool, default=True
        Whether to use highly variable genes only. If True and use_rep is None,
        uses adata.X[:, adata.var.highly_variable] if available.
        
    key_added : str, optional
        Key under which to add the computed distances and connectivities.
        If None, uses 'mutual_neighbors'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to MutualNearestNeighbors.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
    """
    if key_added is None:
        key_added = 'mutual_neighbors'
    
    adata = adata.copy() if copy else adata
    
    # Extract data matrix
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        if use_rep is not None:
            if use_rep == 'X':
                X = adata.X
            else:
                X = adata.obsm[use_rep]
        else:
            if adata.n_vars < 50:
                X = adata.X
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                X = adata.obsm['X_pca']
            else:
                X = adata.X
        
        # Apply highly variable genes filter if requested and applicable
        if use_highly_variable and use_rep == "X" and 'highly_variable' in adata.var.columns:
            if sparse.issparse(X):
                X = X[:, adata.var.highly_variable.values].toarray()
            else:
                X = X[:, adata.var.highly_variable.values]
        elif sparse.issparse(X):
            X = X.toarray()
    else:
        X = adata
    
    # Compute mutual neighbors
    mnn = MutualNearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        **kwargs
    )
    mnn.fit(X)
    
    # Get distance matrix from fitted model
    distance_matrix = mnn.matrix_
    
    # Convert to connectivities using treeclust's original implementation
    # This preserves the intended sparsity for mutual neighbors
    connectivities = to_connectivities_matrix(distance_matrix, k_neighbors=n_neighbors)
    transitions = to_transition_matrix(distance_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        connectivities_key = f'{key_added}_connectivities' if key_added != 'mutual_neighbors' else 'mutual_connectivities'
        distances_key = f'{key_added}_distances' if key_added != 'mutual_neighbors' else 'mutual_distances'
        
        adata.obsp[connectivities_key] = connectivities
        adata.obsp[distances_key] = distance_matrix
        
        adata.uns[key_added] = {
            'connectivities_key': connectivities_key,
            'distances_key': distances_key,
            'params': {
                'n_neighbors': n_neighbors,
                'method': 'mutual',
                'metric': metric
            },
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    else:
        return {
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    
    return adata if copy else None


def consensus_mutual_neighbors(
    adata: Union['anndata.AnnData', np.ndarray],
    n_neighbors: int = 15,
    n_splits: int = 10,
    metric: str = 'euclidean',
    use_rep: Optional[str] = None,
    use_highly_variable: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
    random_state: Optional[int] = 0,
    pipeline_bootstrapper: Optional[Any] = None,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Compute consensus mutual nearest neighbors graph.
    
    Combines consensus sampling with mutual neighbors for maximum
    robustness and sparsity. Uses treeclust's original connectivity
    computation to preserve intended behavior.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or data matrix.
        
    n_neighbors : int, default=15
        Number of nearest neighbors to compute initially.
        
    n_splits : int, default=10
        Number of bootstrap samples for consensus.
        
    metric : str, default='euclidean'
        Distance metric to use for neighborhood computation.
        
    use_rep : str, optional
        Use the indicated representation. If None, uses .X.
        
    use_highly_variable : bool, default=True
        Whether to use highly variable genes only. If True and use_rep is None,
        uses adata.X[:, adata.var.highly_variable] if available.
        
    key_added : str, optional
        Key under which to add the computed distances and connectivities.
        If None, uses 'consensus_mutual_neighbors'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    random_state : int, optional
        Random seed for reproducible results. Controls the randomness of the
        bootstrap sampling and ensures consistent results across runs.
        
    pipeline_bootstrapper : PipelineBootstrapper, optional
        Custom PipelineBootstrapper instance for controlling the bootstrap process.
        If None, creates a default PipelineBootstrapper with 80% sample ratio.
        
    **kwargs
        Additional arguments passed to ConsensusMutualNearestNeighbors.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
    """
    if key_added is None:
        key_added = 'consensus_mutual_neighbors'
    
    adata = adata.copy() if copy else adata
    
    # Extract data matrix
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        if use_rep is not None:
            if use_rep == 'X':
                X = adata.X
            else:
                X = adata.obsm[use_rep]
        else:
            if adata.n_vars < 50:
                X = adata.X
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                X = adata.obsm['X_pca']
            else:
                X = adata.X
        
        # Apply highly variable genes filter if requested and applicable
        if use_highly_variable and use_rep == "X" and 'highly_variable' in adata.var.columns:
            if sparse.issparse(X):
                X = X[:, adata.var.highly_variable.values].toarray()
            else:
                X = X[:, adata.var.highly_variable.values]
        elif sparse.issparse(X):
            X = X.toarray()
    else:
        X = adata
    
    # Compute consensus mutual neighbors
    cmnn = ConsensusMutualNearestNeighbors(
        n_neighbors=n_neighbors,
        n_splits=n_splits,
        metric=metric,
        random_state=random_state,
        pipeline_bootstrapper=pipeline_bootstrapper,
        **kwargs
    )
    cmnn.fit(X)
    
    # Get distance matrix from fitted model
    distance_matrix = cmnn.matrix_
    
    # Convert to connectivities using treeclust's original implementation
    connectivities = to_connectivities_matrix(distance_matrix, k_neighbors=n_neighbors)
    transitions = to_transition_matrix(distance_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        connectivities_key = f'{key_added}_connectivities' if key_added != 'consensus_mutual_neighbors' else 'consensus_mutual_connectivities'
        distances_key = f'{key_added}_distances' if key_added != 'consensus_mutual_neighbors' else 'consensus_mutual_distances'
        
        adata.obsp[connectivities_key] = connectivities
        adata.obsp[distances_key] = distance_matrix
        
        adata.uns[key_added] = {
            'connectivities_key': connectivities_key,
            'distances_key': distances_key,
            'params': {
                'n_neighbors': n_neighbors,
                'n_splits': n_splits,
                'method': 'consensus_mutual',
                'metric': metric
            },
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    else:
        return {
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    
    return adata if copy else None


def coassociation_matrix(
    adata: Union['anndata.AnnData', np.ndarray],
    n_splits: int = 10,
    metric: str = 'euclidean',
    use_rep: Optional[str] = None,
    use_highly_variable: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs
) -> Optional['anndata.AnnData']:
    """
    Compute coassociation distance matrix for neighborhood analysis.
    
    Measures how often pairs of points appear in the same clusters
    across multiple bootstrap samples. Uses treeclust's original
    connectivity computation.
    
    Parameters
    ----------
    adata : AnnData or np.ndarray
        Annotated data object or data matrix.
        
    n_splits : int, default=10
        Number of bootstrap samples.
        
    metric : str, default='euclidean'
        Distance metric to use for neighborhood computation.
        
    use_rep : str, optional
        Use the indicated representation. If None, uses .X.
        
    use_highly_variable : bool, default=True
        Whether to use highly variable genes only. If True and use_rep is None,
        uses adata.X[:, adata.var.highly_variable] if available.
        
    key_added : str, optional
        Key under which to add the computed distances and connectivities.
        If None, uses 'coassociation'.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    **kwargs
        Additional arguments passed to CoassociationDistanceMatrix.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy.
    """
    if key_added is None:
        key_added = 'coassociation'
    
    adata = adata.copy() if copy else adata
    
    # Extract data matrix
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        if use_rep is not None:
            if use_rep == 'X':
                X = adata.X
            else:
                X = adata.obsm[use_rep]
        else:
            if adata.n_vars < 50:
                X = adata.X
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                X = adata.obsm['X_pca']
            else:
                X = adata.X
        
        # Apply highly variable genes filter if requested and applicable
        if use_highly_variable and use_rep == "X" and 'highly_variable' in adata.var.columns:
            if sparse.issparse(X):
                X = X[:, adata.var.highly_variable.values].toarray()
            else:
                X = X[:, adata.var.highly_variable.values]
        elif sparse.issparse(X):
            X = X.toarray()
    else:
        X = adata
    
    # Create a simple KMeans clustering for coassociation
    from sklearn.cluster import KMeans
    
    # Compute coassociation matrix with KMeans clustering
    cam = CoassociationDistanceMatrix(
        clustering_classes=KMeans(n_clusters=10, random_state=42),
        n_splits=n_splits,
        **kwargs
    )
    cam.fit(X)
    
    # Get distance matrix from fitted model
    distance_matrix = cam.distance_matrix_
    
    # For coassociation, we might want to derive neighbors from the distance matrix
    # Let's use a reasonable number of neighbors (15 by default)
    n_neighbors = 15
    
    # Convert to connectivities using treeclust's original implementation
    connectivities = to_connectivities_matrix(distance_matrix, k_neighbors=n_neighbors)
    transitions = to_transition_matrix(distance_matrix)
    
    # Store results
    if HAS_ANNDATA and isinstance(adata, anndata.AnnData):
        connectivities_key = f'{key_added}_connectivities' if key_added != 'coassociation' else 'coassociation_connectivities'
        distances_key = f'{key_added}_distances' if key_added != 'coassociation' else 'coassociation_distances'
        
        adata.obsp[connectivities_key] = connectivities
        adata.obsp[distances_key] = distance_matrix
        
        adata.uns[key_added] = {
            'connectivities_key': connectivities_key,
            'distances_key': distances_key,
            'params': {
                'n_neighbors': n_neighbors,
                'n_splits': n_splits,
                'method': 'coassociation',
                'metric': metric
            },
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    else:
        return {
            'distances': distance_matrix,
            'connectivities': connectivities,
            'transitions': transitions
        }
    
    return adata if copy else None