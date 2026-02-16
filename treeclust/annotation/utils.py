"""
Utility functions for treeclust.

This module contains various utility functions used throughout the treeclust package,
including label propagation and other helper functions.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, Dict, Optional, Any, Tuple

def propagate_annotation(
    matrix: Union[np.ndarray, sparse.spmatrix],
    annotations: pd.DataFrame,
    k: int = 1,
    normalize: bool = True,
    alpha: float = 0.9
) -> pd.DataFrame:
    """
    Propagate annotations through a matrix using transition matrix-based label propagation.
    
    This function treats the input matrix as a transition matrix and propagates 
    annotations through it. The annotations DataFrame should have the same number of
    rows as the matrix dimensions, with columns representing different annotation classes.
    
    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse matrix
        Connectivity/similarity matrix of shape (n_nodes, n_nodes).
        Will be normalized to create a transition matrix.
        
    annotations : pd.DataFrame
        Annotation data with shape (n_nodes, n_classes).
        Each row corresponds to a node/observation in the matrix.
        Each column represents a different annotation class or probability.
        Can contain binary indicators (0/1) or probability values (0.0-1.0).
        
    k : int, default=1
        Number of propagation steps. Higher values spread labels further.
        
    normalize : bool, default=True
        Whether to normalize the matrix rows to sum to 1 (transition matrix).
        If False, uses the matrix as-is.
        
    alpha : float, default=0.9
        Damping factor for iterative propagation. Only used if k > 1.
        Controls how much the original annotations are preserved vs propagated labels.
        alpha=1.0 means only propagation, alpha=0.0 means only original annotations.
        
    Returns
    -------
    propagated : pd.DataFrame
        Propagated annotations of shape (n_nodes, n_classes).
        Same structure as input annotations DataFrame but with propagated values.
        
    Examples
    --------
    # Basic annotation propagation
    >>> import pandas as pd
    >>> adjacency = np.array([[0, 1, 1, 0, 0],
    ...                       [1, 0, 1, 1, 0], 
    ...                       [1, 1, 0, 1, 1],
    ...                       [0, 1, 1, 0, 1],
    ...                       [0, 0, 1, 1, 0]])
    >>> annotations = pd.DataFrame({
    ...     'T_cell': [1.0, 0.0, 0.0, 0.0, 0.0],
    ...     'B_cell': [0.0, 1.0, 0.0, 0.0, 0.0],
    ...     'unknown': [0.0, 0.0, 1.0, 1.0, 1.0]
    ... })
    >>> propagated = propagate_annotation(adjacency, annotations)
    >>> print(propagated.head())
    
    # Multi-step propagation
    >>> propagated_k3 = propagate_annotation(adjacency, annotations, k=3)
    
    # With probability annotations
    >>> prob_annotations = pd.DataFrame({
    ...     'class_A': [1.0, 0.0, 0.5, 0.0, 0.0],
    ...     'class_B': [0.0, 1.0, 0.5, 0.0, 0.0]
    ... })
    >>> propagated = propagate_annotation(adjacency, prob_annotations)
    """
    # Validate input dimensions
    if not isinstance(annotations, pd.DataFrame):
        raise TypeError("annotations must be a pandas DataFrame")
    
    # Convert matrix to dense array for easier manipulation
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)
    
    n_nodes = matrix.shape[0]
    
    # Check that annotations has the same number of rows as matrix dimensions
    if len(annotations) != n_nodes:
        raise ValueError(f"Number of annotations ({len(annotations)}) must match "
                        f"matrix dimensions ({n_nodes})")
    
    # Convert annotations to numpy array for computation
    annotations_array = annotations.values.astype(float)
    n_classes = annotations_array.shape[1]
    
    # Normalize matrix to transition matrix if requested
    if normalize:
        transition_matrix = matrix.copy()
        row_sums = np.sum(transition_matrix, axis=1)
        # Avoid division by zero
        non_zero_mask = row_sums > 0
        transition_matrix[non_zero_mask] = transition_matrix[non_zero_mask] / row_sums[non_zero_mask, np.newaxis]
    else:
        transition_matrix = matrix.copy()
    
    # Propagate annotations
    propagated = annotations_array.copy()
    
    if k == 1:
        # Single step propagation
        propagated = transition_matrix @ propagated
    else:
        # Multi-step iterative propagation with damping
        original_annotations = annotations_array.copy()
        
        for step in range(k):
            # Propagate one step
            new_propagated = transition_matrix @ propagated
            
            # Apply damping factor to preserve original annotations
            propagated = alpha * new_propagated + (1 - alpha) * original_annotations
    
    # Return as DataFrame with same structure as input
    result_df = pd.DataFrame(
        propagated,
        index=annotations.index,
        columns=annotations.columns
    )
    
    return result_df


def propagate_labels_iterative(
    matrix: Union[np.ndarray, sparse.spmatrix],
    annotations: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
    max_iter: int = 100,
    tolerance: float = 1e-6,
    alpha: float = 0.9
) -> tuple:
    """
    Iterative annotation propagation until convergence.
    
    Runs annotation propagation until the change in annotations falls below tolerance
    or maximum iterations are reached. Supports both numpy arrays and pandas DataFrames.
    
    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse matrix
        Connectivity/similarity matrix of shape (n_nodes, n_nodes).
        
    annotations : np.ndarray, scipy.sparse matrix, or pd.DataFrame
        Initial annotation data. If DataFrame, preserves column names and index.
        
    max_iter : int, default=100
        Maximum number of iterations.
        
    tolerance : float, default=1e-6
        Convergence tolerance (max change in probabilities).
        
    alpha : float, default=0.9
        Damping factor for preserving original annotations.
        
    Returns
    -------
    propagated : np.ndarray or pd.DataFrame
        Final propagated annotations. Type matches input annotations type.
        
    n_iterations : int
        Number of iterations performed.
        
    converged : bool
        Whether convergence was achieved.
        
    Examples
    --------
    >>> labels = np.array([0, 1, -1, -1, 0])  
    >>> adjacency = np.array([[0, 1, 1, 0, 0],
    ...                       [1, 0, 1, 1, 0], 
    ...                       [1, 1, 0, 1, 1],
    ...                       [0, 1, 1, 0, 1],
    ...                       [0, 0, 1, 1, 0]])
    >>> propagated, n_iter, converged = propagate_labels_iterative(adjacency, labels)
    >>> print(f"Converged after {n_iter} iterations: {converged}")
    
    >>> # With DataFrame
    >>> import pandas as pd
    >>> annotations = pd.DataFrame({'T_cell': [1, 0, 0, 0, 0], 'B_cell': [0, 1, 0, 0, 0]})
    >>> propagated_df, n_iter, converged = propagate_labels_iterative(adjacency, annotations)
    """
    # Convert annotations to DataFrame if needed
    if not isinstance(annotations, pd.DataFrame):
        if sparse.issparse(annotations):
            annotations = annotations.toarray()
        annotations_array = np.asarray(annotations)
        
        # Handle 1D arrays by converting to DataFrame
        if annotations_array.ndim == 1:
            # Convert 1D labels to one-hot DataFrame
            unique_labels = np.unique(annotations_array)
            if -1 in unique_labels:
                valid_labels = unique_labels[unique_labels != -1]
            elif np.any(np.isnan(unique_labels)):
                valid_labels = unique_labels[~np.isnan(unique_labels)]
            else:
                valid_labels = unique_labels
            
            # Create column names for each class
            class_names = [f'class_{label}' for label in valid_labels]
            label_to_idx = {label: i for i, label in enumerate(valid_labels)}
            
            # Create one-hot matrix
            n_nodes = len(annotations_array)
            n_classes = len(valid_labels)
            one_hot = np.zeros((n_nodes, n_classes))
            for i, label in enumerate(annotations_array):
                if label in label_to_idx:
                    one_hot[i, label_to_idx[label]] = 1.0
            
            annotations_df = pd.DataFrame(one_hot, columns=class_names)
        else:
            # 2D array - create DataFrame with generic column names
            n_classes = annotations_array.shape[1]
            class_names = [f'class_{i}' for i in range(n_classes)]
            annotations_df = pd.DataFrame(annotations_array, columns=class_names)
    else:
        annotations_df = annotations.copy()
    
    # Initial propagation
    propagated = propagate_annotation(matrix, annotations_df, k=1, alpha=1.0)
    
    # Convert to numpy for iteration (but keep original DataFrame structure)
    original_annotations = annotations_df.values.copy()
    propagated_array = propagated.values.copy()
    
    # Normalize matrix
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)
    
    transition_matrix = matrix.copy()
    row_sums = np.sum(transition_matrix, axis=1)
    non_zero_mask = row_sums > 0
    transition_matrix[non_zero_mask] = transition_matrix[non_zero_mask] / row_sums[non_zero_mask, np.newaxis]
    
    # Iterative propagation
    for iteration in range(max_iter):
        previous_propagated = propagated_array.copy()
        
        # One propagation step
        new_propagated = transition_matrix @ propagated_array
        
        # Apply damping
        propagated_array = alpha * new_propagated + (1 - alpha) * original_annotations
        
        # Check convergence
        max_change = np.max(np.abs(propagated_array - previous_propagated))
        if max_change < tolerance:
            # Return as DataFrame if input was DataFrame, otherwise as numpy array
            if isinstance(annotations, pd.DataFrame):
                result_df = pd.DataFrame(propagated_array, index=annotations.index, columns=annotations.columns)
                return result_df, iteration + 1, True
            else:
                return propagated_array, iteration + 1, True
    
    # Return final result
    if isinstance(annotations, pd.DataFrame):
        result_df = pd.DataFrame(propagated_array, index=annotations.index, columns=annotations.columns)
        return result_df, max_iter, False
    else:
        return propagated_array, max_iter, False

def annotation_to_hierarchical_clustering_projection(
    annotation_vector: Union[np.ndarray, pd.Series],
    labels_dict: Dict[Any, np.ndarray],
    metric: str = 'jaccard',
    min_overlap: float = 0.0,
    return_scores: bool = False
) -> Union[Dict[Any, Dict], Tuple[Dict[Any, Dict], Dict[Any, Dict]]]:
    """
    Find the best cluster correspondence between an annotation vector and hierarchical clustering.
    
    For each annotation class, finds the cluster at each resolution that has the highest
    similarity according to the specified metric. This is useful for mapping external
    annotations (e.g., cell types) to clustering results.
    
    Parameters
    ----------
    annotation_vector : np.ndarray or pd.Series
        Vector of annotations/labels for each cell. Can contain strings, integers,
        or any hashable values. Missing/unknown values can be represented as None, NaN,
        or a special string like 'unknown'.
        
    labels_dict : dict
        Dictionary mapping resolution indices/names to cluster label arrays.
        Example: {0: array([0, 0, 1, 1, 2]), 1: array([0, 1, 1, 2, 2])}
        
    metric : str, default='jaccard'
        Similarity metric to use for comparing annotation classes to clusters:
        - 'jaccard': Jaccard similarity (intersection over union)
        - 'overlap': Simple overlap coefficient (intersection over min set size)
        - 'precision': Precision (true positives / (true positives + false positives))
        - 'recall': Recall (true positives / (true positives + false negatives))
        - 'f1': F1 score (harmonic mean of precision and recall)
        
    min_overlap : float, default=0.0
        Minimum similarity score to report a correspondence. Correspondences
        with similarity below this threshold will be excluded from results.
        
    return_scores : bool, default=False
        If True, returns both correspondences and similarity scores.
        If False, returns only correspondences.
        
    Returns
    -------
    correspondences : dict
        Dictionary mapping annotation classes to their best cluster matches.
        Format: {annotation_class: {resolution: (cluster_id, similarity_score)}}
        
    scores : dict, optional (if return_scores=True)
        Dictionary with full similarity matrices for each annotation class.
        Format: {annotation_class: {resolution: {cluster_id: similarity_score}}}
        
    Examples
    --------
    >>> # Basic usage with cell type annotations
    >>> annotations = np.array(['T_cell', 'T_cell', 'B_cell', 'B_cell', 'NK_cell'])
    >>> labels = {
    ...     0: np.array([0, 0, 1, 1, 2]),  # 3 clusters at resolution 0
    ...     1: np.array([0, 1, 2, 2, 3])   # 4 clusters at resolution 1  
    ... }
    >>> correspondences = find_cluster_correspondence(annotations, labels)
    >>> print(correspondences)
    # {'T_cell': {0: (0, 1.0), 1: (0, 0.5)}, 
    #  'B_cell': {0: (1, 1.0), 1: (2, 1.0)}, 
    #  'NK_cell': {0: (2, 1.0), 1: (3, 1.0)}}
    
    >>> # With different metrics
    >>> corr_f1 = find_cluster_correspondence(annotations, labels, metric='f1')
    >>> corr_precision = find_cluster_correspondence(annotations, labels, metric='precision')
    
    >>> # Get similarity scores too
    >>> correspondences, scores = find_cluster_correspondence(
    ...     annotations, labels, return_scores=True
    ... )
    >>> print(scores['T_cell'][0])  # Similarity of T_cell to all clusters in resolution 0
    
    >>> # Filter low similarities
    >>> high_conf_corr = find_cluster_correspondence(
    ...     annotations, labels, min_overlap=0.5
    ... )
    """
    
    # Convert annotation vector to numpy array
    if isinstance(annotation_vector, pd.Series):
        annotations = annotation_vector.values
    else:
        annotations = np.asarray(annotation_vector)
    
    # Validate inputs
    if not labels_dict:
        raise ValueError("labels_dict cannot be empty")
    
    # Get number of cells and validate label arrays
    n_cells = len(annotations)
    for res_key, labels in labels_dict.items():
        if labels is not None and len(labels) != n_cells:
            raise ValueError(f"All label arrays must have same length. "
                           f"Resolution {res_key} has {len(labels)} labels, expected {n_cells}")
    
    # Get unique annotation classes (exclude None, NaN, and common missing value indicators)
    valid_annotations = []
    for ann in annotations:
        if ann is not None and not (isinstance(ann, float) and np.isnan(ann)):
            if not (isinstance(ann, str) and ann.lower() in ['unknown', 'unassigned', 'na', 'nan', '']):
                valid_annotations.append(ann)
    
    unique_classes = sorted(set(valid_annotations))
    
    if not unique_classes:
        raise ValueError("No valid annotation classes found")
    
    # Initialize results
    correspondences = {}
    all_scores = {} if return_scores else None
    
    # Calculate similarities for each annotation class
    for ann_class in unique_classes:
        correspondences[ann_class] = {}
        if return_scores:
            all_scores[ann_class] = {}
        
        # Create mask for cells with this annotation
        ann_mask = (annotations == ann_class)
        ann_indices = set(np.where(ann_mask)[0])
        
        # Compare against clusters in each resolution
        for res_key, labels in labels_dict.items():
            if labels is None:
                continue
                
            labels_array = np.asarray(labels)
            unique_clusters = sorted(set(labels_array))
            
            best_cluster = None
            best_score = -1.0
            cluster_scores = {}
            
            # Calculate similarity to each cluster
            for cluster_id in unique_clusters:
                cluster_mask = (labels_array == cluster_id)
                cluster_indices = set(np.where(cluster_mask)[0])
                
                # Calculate similarity based on chosen metric
                similarity = _calculate_similarity(ann_indices, cluster_indices, metric)
                cluster_scores[cluster_id] = similarity
                
                # Track best match
                if similarity > best_score:
                    best_score = similarity
                    best_cluster = cluster_id
            
            # Store results if they meet minimum threshold
            if best_score >= min_overlap:
                correspondences[ann_class][res_key] = (best_cluster, best_score)
            
            if return_scores:
                all_scores[ann_class][res_key] = cluster_scores
    
    if return_scores:
        return correspondences, all_scores
    else:
        return correspondences


def _calculate_similarity(set1: set, set2: set, metric: str) -> float:
    """Calculate similarity between two sets of indices using specified metric."""
    
    intersection_size = len(set1.intersection(set2))
    
    if metric == 'jaccard':
        union_size = len(set1.union(set2))
        return intersection_size / union_size if union_size > 0 else 0.0
        
    elif metric == 'overlap':
        min_size = min(len(set1), len(set2))
        return intersection_size / min_size if min_size > 0 else 0.0
        
    elif metric == 'precision':
        # Precision: TP / (TP + FP) = intersection / |set2|
        return intersection_size / len(set2) if len(set2) > 0 else 0.0
        
    elif metric == 'recall':
        # Recall: TP / (TP + FN) = intersection / |set1|
        return intersection_size / len(set1) if len(set1) > 0 else 0.0
        
    elif metric == 'f1':
        # F1 score: harmonic mean of precision and recall
        precision = intersection_size / len(set2) if len(set2) > 0 else 0.0
        recall = intersection_size / len(set1) if len(set1) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        else:
            return 2 * (precision * recall) / (precision + recall)
    
    else:
        raise ValueError(f"Unknown metric: {metric}. "
                        f"Supported metrics: 'jaccard', 'overlap', 'precision', 'recall', 'f1'")


def find_annotation_cluster_match(
    annotation_vector: Union[np.ndarray, pd.Series],
    cluster_labels: np.ndarray,
    metric: str = 'jaccard',
    min_overlap: float = 0.0,
    return_scores: bool = False
) -> Union[Dict[Any, Tuple], Tuple[Dict[Any, Tuple], Dict[Any, float]]]:
    """
    Find the best cluster match for each annotation class in a single clustering.
    
    This is a simplified, non-hierarchical version of find_cluster_correspondence()
    that works with a single set of cluster labels. For each annotation class,
    finds the cluster that has the highest similarity according to the specified metric.
    
    Parameters
    ----------
    annotation_vector : np.ndarray or pd.Series
        Vector of annotations/labels for each cell. Can contain strings, integers,
        or any hashable values. Missing/unknown values can be represented as None, NaN,
        or a special string like 'unknown'.
        
    cluster_labels : np.ndarray
        Array of cluster labels for each cell. Should have the same length as
        annotation_vector.
        
    metric : str, default='jaccard'
        Similarity metric to use for comparing annotation classes to clusters:
        - 'jaccard': Jaccard similarity (intersection over union)
        - 'overlap': Simple overlap coefficient (intersection over min set size)
        - 'precision': Precision (true positives / (true positives + false positives))
        - 'recall': Recall (true positives / (true positives + false negatives))
        - 'f1': F1 score (harmonic mean of precision and recall)
        
    min_overlap : float, default=0.0
        Minimum similarity score to report a correspondence. Correspondences
        with similarity below this threshold will be excluded from results.
        
    return_scores : bool, default=False
        If True, returns both correspondences and all similarity scores.
        If False, returns only correspondences.
        
    Returns
    -------
    correspondences : dict
        Dictionary mapping annotation classes to their best cluster matches.
        Format: {annotation_class: (cluster_id, similarity_score)}
        
    all_scores : dict, optional (if return_scores=True)
        Dictionary with similarity scores for each annotation class to all clusters.
        Format: {annotation_class: {cluster_id: similarity_score}}
        
    Examples
    --------
    >>> # Basic usage with cell type annotations
    >>> annotations = np.array(['T_cell', 'T_cell', 'B_cell', 'B_cell', 'NK_cell'])
    >>> clusters = np.array([0, 0, 1, 1, 2])
    >>> correspondences = find_annotation_cluster_match(annotations, clusters)
    >>> print(correspondences)
    # {'T_cell': (0, 1.0), 'B_cell': (1, 1.0), 'NK_cell': (2, 1.0)}
    
    >>> # With different metrics
    >>> corr_f1 = find_annotation_cluster_match(annotations, clusters, metric='f1')
    >>> corr_precision = find_annotation_cluster_match(annotations, clusters, metric='precision')
    
    >>> # Get all similarity scores
    >>> correspondences, scores = find_annotation_cluster_match(
    ...     annotations, clusters, return_scores=True
    ... )
    >>> print(scores['T_cell'])  # Similarity of T_cell to all clusters
    # {0: 1.0, 1: 0.0, 2: 0.0}
    
    >>> # Filter low similarities
    >>> high_conf_corr = find_annotation_cluster_match(
    ...     annotations, clusters, min_overlap=0.5
    ... )
    
    >>> # With pandas Series
    >>> import pandas as pd
    >>> ann_series = pd.Series(annotations, name='cell_type')
    >>> correspondences = find_annotation_cluster_match(ann_series, clusters)
    """
    
    # Convert inputs to numpy arrays
    if isinstance(annotation_vector, pd.Series):
        annotations = annotation_vector.values
    else:
        annotations = np.asarray(annotation_vector)
    
    cluster_labels = np.asarray(cluster_labels)
    
    # Validate inputs
    if len(annotations) != len(cluster_labels):
        raise ValueError(f"annotation_vector and cluster_labels must have same length. "
                        f"Got {len(annotations)} and {len(cluster_labels)}")
    
    # Get unique annotation classes (exclude None, NaN, and common missing value indicators)
    valid_annotations = []
    for ann in annotations:
        if ann is not None and not (isinstance(ann, float) and np.isnan(ann)):
            if not (isinstance(ann, str) and ann.lower() in ['unknown', 'unassigned', 'na', 'nan', '']):
                valid_annotations.append(ann)
    
    unique_classes = sorted(set(valid_annotations))
    unique_clusters = sorted(set(cluster_labels))
    
    if not unique_classes:
        raise ValueError("No valid annotation classes found")
    
    if not unique_clusters:
        raise ValueError("No clusters found in cluster_labels")
    
    # Initialize results
    correspondences = {}
    all_scores = {} if return_scores else None
    
    # Calculate similarities for each annotation class
    for ann_class in unique_classes:
        # Create mask for cells with this annotation
        ann_mask = (annotations == ann_class)
        ann_indices = set(np.where(ann_mask)[0])
        
        best_cluster = None
        best_score = -1.0
        cluster_scores = {}
        
        # Calculate similarity to each cluster
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = set(np.where(cluster_mask)[0])
            
            # Calculate similarity based on chosen metric
            similarity = _calculate_similarity(ann_indices, cluster_indices, metric)
            cluster_scores[cluster_id] = similarity
            
            # Track best match
            if similarity > best_score:
                best_score = similarity
                best_cluster = cluster_id
        
        # Store results if they meet minimum threshold
        if best_score >= min_overlap:
            correspondences[ann_class] = (best_cluster, best_score)
        
        if return_scores:
            all_scores[ann_class] = cluster_scores
    
    if return_scores:
        return correspondences, all_scores
    else:
        return correspondences


def annotation_hierarchical(
    labels_dict: Dict[Any, np.ndarray],
    annotations: Dict[Tuple[Any, Any], Union[str, int, float]], 
    weights_dict: Optional[Dict[Any, np.ndarray]] = None,
    default_value: Optional[Union[str, int, float]] = None,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Create hierarchical annotations for cells based on cluster labels and weights.
    
    This function takes hierarchical clustering results (labels at different resolutions)
    and creates soft annotations for all cells based on cluster-to-annotation mappings.
    It uses confidence weights to handle cases where cells belong to multiple 
    annotated clusters across different resolutions.
    
    Parameters
    ----------
    labels_dict : dict
        Dictionary mapping resolution indices/names to cluster label arrays.
        Example: {0: array([0, 0, 1, 1, 2]), 1: array([0, 1, 1, 2, 2])}
        
    annotations : dict
        Dictionary mapping (resolution_key, cluster_id) tuples to annotation values.
        Example: {(0, 1): 'T_cell', (0, 2): 'B_cell', (1, 3): 'T_cell'}
        
    weights_dict : dict, optional
        Dictionary mapping resolution indices/names to weight arrays for each cell.
        Weights represent confidence/consensus scores. If None, uses uniform weights.
        Example: {0: array([0.9, 0.9, 0.8, 0.8, 0.7]), 1: array([0.8, 0.9, 0.9, 0.6, 0.6])}
        
    default_value : str, int, float, or None, optional
        Value to assign to unassigned class. If None, automatically determined
        from annotation types ('unassigned' for strings, np.nan for numbers).
        
    normalize : bool, default=True
        Whether to normalize probabilities so each cell sums to 1.
        If False, returns raw weighted scores.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cells as rows and annotation classes as columns.
        Values represent probabilities/weights for each cell-class combination.
        Includes an 'unassigned' class for cells with no or weak annotations.
        
    Examples
    --------
    >>> # Basic hierarchical annotation
    >>> labels = {
    ...     0: np.array([0, 0, 1, 1, 2]),  # 3 clusters at resolution 0
    ...     1: np.array([0, 1, 1, 2, 2])   # 3 clusters at resolution 1  
    ... }
    >>> annotations = {
    ...     (0, 1): 'T_cell',      # Resolution 0, cluster 1 = T cells
    ...     (0, 2): 'B_cell',      # Resolution 0, cluster 2 = B cells
    ...     (1, 0): 'NK_cell'      # Resolution 1, cluster 0 = NK cells
    ... }
    >>> result_df = annotation_hierarchical(labels, annotations)
    >>> print(result_df.columns.tolist())  # ['B_cell', 'NK_cell', 'T_cell', 'unassigned']
    >>> print(result_df.shape)  # (5, 4) - 5 cells, 4 classes
    
    >>> # With custom weights (e.g., from consensus matrices)
    >>> weights = {
    ...     0: np.array([0.9, 0.9, 0.8, 0.8, 0.7]),  # High confidence for res 0
    ...     1: np.array([0.6, 0.7, 0.9, 0.8, 0.8])   # Variable confidence for res 1
    ... }
    >>> result_df = annotation_hierarchical(labels, annotations, weights)
    
    >>> # Without normalization (raw weighted scores)
    >>> raw_scores = annotation_hierarchical(labels, annotations, weights, normalize=False)
    """
    
    # Validate inputs
    if not labels_dict:
        raise ValueError("labels_dict cannot be empty")
    if not annotations:
        raise ValueError("annotations cannot be empty")
    
    # Get number of cells from first labels array
    n_cells = None
    for labels in labels_dict.values():
        if labels is not None:
            n_cells = len(labels)
            break
    
    if n_cells is None:
        raise ValueError("No valid labels found in labels_dict")
    
    # Validate all label arrays have same length
    for res_key, labels in labels_dict.items():
        if labels is not None and len(labels) != n_cells:
            raise ValueError(f"All label arrays must have same length. "
                           f"Resolution {res_key} has {len(labels)} labels, expected {n_cells}")
    
    # Determine default value if not provided
    if default_value is None:
        if annotations:
            sample_value = next(iter(annotations.values()))
            if isinstance(sample_value, str):
                default_value = 'unassigned'
            else:
                default_value = np.nan
        else:
            default_value = 'unassigned'
    
    # Get unique class names and create mapping
    class_names = sorted(set(annotations.values()))
    if default_value not in class_names:
        class_names.append(default_value)
    
    n_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Initialize probability matrix
    probabilities = np.zeros((n_cells, n_classes))
    
    # Process each annotation
    for (res_key, cluster_id), label in annotations.items():
        if res_key not in labels_dict:
            print(f"Warning: Resolution {res_key} not found in labels_dict, skipping")
            continue
            
        labels = labels_dict[res_key]
        if labels is None:
            print(f"Warning: No labels found for resolution {res_key}, skipping")
            continue
            
        # Find cells in this cluster
        labels_array = np.array(labels)
        cluster_mask = (labels_array == cluster_id)
        if not np.any(cluster_mask):
            print(f"Warning: Cluster {cluster_id} not found in resolution {res_key}, skipping")
            continue
        
        # Get weights for this resolution (use uniform if not provided)
        if weights_dict is not None and res_key in weights_dict:
            weights = np.array(weights_dict[res_key])
            if len(weights) != n_cells:
                raise ValueError(f"Weights for resolution {res_key} have length {len(weights)}, "
                               f"expected {n_cells}")
        else:
            weights = np.ones(n_cells)
        
        # Apply cluster mask to weights
        weights = weights * cluster_mask.astype(float)
        
        # Add weights to appropriate class column
        class_idx = class_to_idx[label]
        probabilities[:, class_idx] += weights
    
    # Calculate unassigned probabilities: max(1 - sum(other_probabilities), 0)
    if default_value in class_to_idx:
        unassigned_idx = class_to_idx[default_value]
        
        # Sum all non-unassigned class probabilities
        other_classes_sum = np.sum(probabilities[:, [i for i in range(n_classes) if i != unassigned_idx]], axis=1)
        
        # Unassigned probability = max(1 - sum(other_probabilities), 0)
        unassigned_probs = np.maximum(1.0 - other_classes_sum, 0.0)
        probabilities[:, unassigned_idx] = unassigned_probs
    
    # Normalize if requested
    if normalize:
        # Normalize each row to sum to 1 (avoid division by zero)
        row_sums = probabilities.sum(axis=1)
        non_zero_mask = row_sums > 0
        probabilities[non_zero_mask] = probabilities[non_zero_mask] / row_sums[non_zero_mask, np.newaxis]
    
    # Create DataFrame
    result_df = pd.DataFrame(
        probabilities, 
        columns=class_names,
        index=pd.RangeIndex(n_cells, name='cell_id')
    )
    
    return result_df