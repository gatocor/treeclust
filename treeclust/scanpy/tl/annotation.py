"""
Scanpy-style annotation tools for treeclust.

This module provides scanpy-compatible annotation functions following the same patterns
as scanpy.tl for seamless integration with AnnData objects.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, Dict, Optional, Any, Tuple, Literal
from ...annotation.utils import propagate_annotation as _propagate_annotation
from ...annotation.utils import annotation_hierarchical as _annotation_hierarchical

try:
    import anndata
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

def annotate(
    adata: 'anndata.AnnData',
    annotation_mapping: Dict[Any, Union[str, int, float]],
    clustering_key: str,
    copy: bool = False,
    key_added: str = 'annotation'
) -> Optional['anndata.AnnData']:
    """
    Assign cell type annotations based on cluster labels.
    
    This function creates cell annotations by mapping cluster identities to cell types
    using a simple dictionary mapping. This is the most basic annotation approach.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    annotation_mapping : dict
        Dictionary mapping cluster IDs to annotation values.
        Example: {0: 'T_cell', 1: 'B_cell', 2: 'NK_cell'}
        
    clustering_key : str
        Key in adata.obs containing the cluster labels to map.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    key_added : str, default='annotation'
        Key under which to add the annotations in adata.obs.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy with annotations.
        
    Examples
    --------
    >>> # Basic cluster-to-annotation mapping
    >>> annotation_map = {0: 'T_cell', 1: 'B_cell', 2: 'NK_cell'}
    >>> tc.tl.annotation(adata, annotation_map, clustering_key='leiden')
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for this function")
    
    adata = adata.copy() if copy else adata
    
    # Get cluster labels
    if clustering_key not in adata.obs.columns:
        raise ValueError(f"Clustering key '{clustering_key}' not found in adata.obs")
    
    cluster_labels = adata.obs[clustering_key]
    
    # Map clusters to annotations
    annotations = cluster_labels.map(annotation_mapping)
    
    # Handle unmapped clusters
    unmapped_mask = annotations.isna()
    if unmapped_mask.any():
        unmapped_clusters = cluster_labels[unmapped_mask].unique()
        print(f"Warning: Unmapped clusters found: {unmapped_clusters}")
        annotations = annotations.fillna('unassigned')
    
    # Store as categorical
    adata.obs[key_added] = pd.Categorical(annotations)
    
    return adata if copy else None

def propagate_annotation(
    adata: 'anndata.AnnData',
    annotation_key: str,
    neighbor_key: Optional[str] = None,
    k: int = 1,
    alpha: float = 0.9,
    copy: bool = False,
    key_added: Optional[str] = None
) -> Optional['anndata.AnnData']:
    """
    Propagate cell type annotations through neighborhood graph.
    
    This function propagates annotations through a connectivity matrix using
    transition matrix-based label propagation. By default, it uses transition
    matrices, but can use any specified neighbor connectivity matrix.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    annotation_key : str
        Key in adata.obs containing the annotations to propagate.
        Can be categorical or contain probability columns.
        
    neighbor_key : str, optional
        Key for the neighbor graph to use. If None, uses 'neighbors' (default).
        The function will look for:
        - {neighbor_key}_connectivities in adata.obsp for connectivity matrix
        - 'transitions' in adata.obsp for transition matrix (default behavior)
        
    k : int, default=1
        Number of propagation steps. Higher values spread labels further.
        
    alpha : float, default=0.9
        Damping factor for iterative propagation. Only used if k > 1.
        Controls how much the original annotations are preserved vs propagated labels.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    key_added : str, optional
        Key under which to add the propagated annotations in adata.obs.
        If None, uses f'{annotation_key}_propagated'.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy with propagated annotations.
        
    Notes
    -----
    The function looks for connectivity matrices in this order:
    1. 'transitions' in adata.obsp (transition matrix - default)
    2. f'{neighbor_key}_connectivities' in adata.obsp
    3. 'connectivities' in adata.obsp (if neighbor_key is None)
    
    Examples
    --------
    >>> # Basic annotation propagation using transition matrix
    >>> import treeclust.scanpy as tc
    >>> tc.tl.annotation(adata, 'cell_type')
    
    >>> # Using specific neighbor graph
    >>> tc.tl.annotation(adata, 'cell_type', neighbor_key='consensus_neighbors')
    
    >>> # Multi-step propagation
    >>> tc.tl.annotation(adata, 'cell_type', k=3, alpha=0.8)
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for this function")
    
    adata = adata.copy() if copy else adata
    
    # Get annotations
    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
    
    annotations = adata.obs[annotation_key]
    
    # Convert to DataFrame format for propagation
    if pd.api.types.is_categorical_dtype(annotations):
        # Convert categorical to one-hot DataFrame
        categories = annotations.cat.categories
        annotation_df = pd.DataFrame(
            np.eye(len(categories))[annotations.cat.codes],
            columns=categories,
            index=annotations.index
        )
        # Handle missing values (-1 codes)
        missing_mask = annotations.cat.codes == -1
        if missing_mask.any():
            annotation_df.loc[missing_mask] = 0
    elif isinstance(annotations, pd.Series):
        # Assume it's a single column - convert to DataFrame
        annotation_df = pd.DataFrame({annotation_key: annotations})
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotations)}")
    
    # Get connectivity matrix - prioritize transition matrix
    matrix = None
    matrix_key = None
    
    # First, try transition matrix (default behavior)
    if 'transitions' in adata.obsp:
        matrix = adata.obsp['transitions']
        matrix_key = 'transitions'
    # Then try specified neighbor key
    elif neighbor_key is not None:
        connectivities_key = f'{neighbor_key}_connectivities'
        if connectivities_key in adata.obsp:
            matrix = adata.obsp[connectivities_key]
            matrix_key = connectivities_key
    # Finally, try default connectivities
    elif 'connectivities' in adata.obsp:
        matrix = adata.obsp['connectivities']
        matrix_key = 'connectivities'
    
    if matrix is None:
        available_keys = list(adata.obsp.keys())
        raise ValueError(
            f"No suitable connectivity matrix found. "
            f"Available keys in adata.obsp: {available_keys}. "
            f"Expected 'transitions', '{neighbor_key}_connectivities' (if neighbor_key provided), "
            f"or 'connectivities'."
        )
    
    # Use transition matrix behavior if we found 'transitions', otherwise normalize
    normalize = (matrix_key != 'transitions')
    
    # Propagate annotations
    propagated_df = _propagate_annotation(
        matrix=matrix,
        annotations=annotation_df,
        k=k,
        normalize=normalize,
        alpha=alpha
    )
    
    # Store results in adata.obs
    if key_added is None:
        key_added = f'{annotation_key}_propagated'
    
    # If original was categorical and we have single dominant class, convert back to categorical
    if pd.api.types.is_categorical_dtype(annotations) and propagated_df.shape[1] > 1:
        # Find dominant class for each cell
        dominant_classes = propagated_df.idxmax(axis=1)
        # Create categorical with same categories as original
        propagated_categorical = pd.Categorical(
            dominant_classes,
            categories=annotations.cat.categories
        )
        adata.obs[key_added] = propagated_categorical
        
        # Also store probability matrix
        for col in propagated_df.columns:
            adata.obs[f'{key_added}_{col}_prob'] = propagated_df[col].values
    else:
        # Store as continuous values
        if propagated_df.shape[1] == 1:
            adata.obs[key_added] = propagated_df.iloc[:, 0].values
        else:
            for col in propagated_df.columns:
                adata.obs[f'{key_added}_{col}'] = propagated_df[col].values
    
    return adata if copy else None

def annotate_hierarchical(
    adata: 'anndata.AnnData',
    annotation_mapping: Dict[Tuple[Any, Any], Union[str, int, float]],
    clustering_key: Optional[str] = None,
    hierarchical_model: Optional[Any] = None,
    confidence_key: Optional[str] = None,
    copy: bool = False,
    key_added: str = 'annotation_hierarchical',
    normalize: bool = True
) -> Optional['anndata.AnnData']:
    """
    Create hierarchical annotations for cells based on cluster labels and confidence scores.
    
    This function creates soft annotations for cells based on hierarchical clustering results
    or multiple clustering resolutions stored in adata.obs. It can use confidence scores
    to weight the annotations appropriately.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
        
    annotation_mapping : dict
        Dictionary mapping (resolution_key, cluster_id) tuples to annotation values.
        Example: {('leiden_0.1', 0): 'T_cell', ('leiden_0.5', 2): 'B_cell'}
        
    clustering_key : str, optional
        Base key for clustering columns in adata.obs. If provided, will look for
        columns like '{clustering_key}_0.1', '{clustering_key}_0.5', etc.
        Cannot be used together with hierarchical_model.
        
    hierarchical_model : object, optional
        A fitted hierarchical clustering model that has a labels_dict attribute
        containing clustering results at different resolutions.
        Cannot be used together with clustering_key.
        
    confidence_key : str, optional
        Key in adata.obs containing confidence scores for each cell.
        If None, uses uniform weights.
        
    copy : bool, default=False
        Whether to return a copy of adata.
        
    key_added : str, default='annotation_hierarchical'
        Key under which to add the hierarchical annotations in adata.obs.
        Individual class probabilities will be stored as '{key_added}_{class_name}'.
        
    normalize : bool, default=True
        Whether to normalize probabilities so each cell sums to 1.
        
    Returns
    -------
    adata : AnnData
        Returns adata if copy=False, otherwise returns a copy with hierarchical annotations.
        
    Notes
    -----
    This function supports two modes:
    1. Using clustering_key: Automatically detects clustering columns in adata.obs
    2. Using hierarchical_model: Uses labels_dict from a fitted hierarchical model
    
    Examples
    --------
    >>> # Using clustering columns in adata.obs
    >>> annotation_map = {
    ...     ('leiden_0.1', 0): 'T_cell',
    ...     ('leiden_0.1', 1): 'B_cell', 
    ...     ('leiden_0.5', 0): 'CD4_T',
    ...     ('leiden_0.5', 1): 'CD8_T'
    ... }
    >>> tc.tl.annotation_hierarchical(
    ...     adata, annotation_map, clustering_key='leiden'
    ... )
    
    >>> # Using hierarchical model
    >>> tc.tl.annotation_hierarchical(
    ...     adata, annotation_map, hierarchical_model=fitted_model
    ... )
    
    >>> # With confidence scores
    >>> tc.tl.annotation_hierarchical(
    ...     adata, annotation_map, clustering_key='leiden', 
    ...     confidence_key='leiden_confidence'
    ... )
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for this function")
    
    # Validate input parameters
    if clustering_key is not None and hierarchical_model is not None:
        raise ValueError("Cannot specify both clustering_key and hierarchical_model")
    
    if clustering_key is None and hierarchical_model is None:
        raise ValueError("Must specify either clustering_key or hierarchical_model")
    
    adata = adata.copy() if copy else adata
    
    # Get labels dictionary
    if hierarchical_model is not None:
        # Use hierarchical model
        if not hasattr(hierarchical_model, 'labels_dict'):
            raise ValueError("hierarchical_model must have a 'labels_dict' attribute")
        labels_dict = hierarchical_model.labels_dict
    else:
        # Use clustering columns from adata.obs
        labels_dict = {}
        clustering_columns = [col for col in adata.obs.columns if col.startswith(clustering_key)]
        
        if not clustering_columns:
            raise ValueError(f"No clustering columns found with base key '{clustering_key}'")
        
        for col in clustering_columns:
            # Extract resolution from column name (e.g., 'leiden_0.5' -> 0.5)
            try:
                resolution_str = col[len(clustering_key) + 1:]  # Remove 'leiden_' part
                if resolution_str.replace('.', '').isdigit():
                    resolution = float(resolution_str)
                else:
                    resolution = col  # Use full column name if not numeric
            except:
                resolution = col  # Fallback to full column name
                
            labels_dict[resolution] = adata.obs[col].values
    
    # Get confidence weights if specified
    weights_dict = None
    if confidence_key is not None:
        if confidence_key not in adata.obs.columns:
            raise ValueError(f"Confidence key '{confidence_key}' not found in adata.obs")
        
        # Create weights_dict with same keys as labels_dict
        confidence_values = adata.obs[confidence_key].values
        weights_dict = {key: confidence_values for key in labels_dict.keys()}
    
    # Create hierarchical annotations
    result_df = _annotation_hierarchical(
        labels_dict=labels_dict,
        annotations=annotation_mapping,
        weights_dict=weights_dict,
        normalize=normalize
    )
    
    # Store results in adata.obs
    for class_name in result_df.columns:
        col_key = f'{key_added}_{class_name}' if len(result_df.columns) > 1 else key_added
        adata.obs[col_key] = result_df[class_name].values
    
    # If there are multiple classes, also create a dominant class assignment
    if result_df.shape[1] > 1:
        dominant_classes = result_df.idxmax(axis=1)
        adata.obs[f'{key_added}_dominant'] = pd.Categorical(dominant_classes)
    
    return adata if copy else None
