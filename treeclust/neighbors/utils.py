"""
Utility functions for neighbor graph processing and matrix conversions.

This module provides helper functions for converting between different
matrix representations commonly used in neighbor graph analysis,
following UMAP-style conventions.
"""

import numpy as np
from scipy import sparse
from typing import Union, Optional, Tuple
import warnings

def to_connectivities_matrix(
    distance_matrix: Union[np.ndarray, sparse.spmatrix],
    k_neighbors: int = 15,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
    metric: str = 'euclidean'
) -> sparse.csr_matrix:
    """
    Convert a distance matrix to UMAP-style connectivity matrix.
    
    Implements UMAP's algorithm for converting k-NN distances to locally
    normalized probabilities using adaptive bandwidth (σᵢ) for each point.
    This creates density-adaptive neighborhoods where each point has roughly
    the same "influence" regardless of local density.
    
    Algorithm:
    1. For each point i, find σᵢ such that Σⱼ exp(-(dᵢⱼ-ρᵢ)/σᵢ) ≈ log₂(k)
    2. Convert distances to probabilities: pᵢⱼ = exp(-(dᵢⱼ-ρᵢ)/σᵢ)  
    3. Symmetrize using fuzzy union: wᵢⱼ = pᵢⱼ + pⱼᵢ - pᵢⱼpⱼᵢ
    
    Parameters
    ----------
    distance_matrix : array-like or sparse matrix
        Input distance matrix (not similarity). Should contain k-NN 
        distances with zeros for non-neighbors.
        
    k_neighbors : int, default=15
        Number of nearest neighbors used to determine local bandwidth.
        Controls the target perplexity for local normalization.
        
    local_connectivity : float, default=1.0
        Number of nearest neighbors that should have probability 1.0.
        Prevents isolated points and ensures local connectivity.
        
    bandwidth : float, default=1.0
        Global bandwidth multiplier. Higher values create broader
        neighborhoods, lower values create tighter neighborhoods.
        
    metric : str, default='euclidean'
        Distance metric (currently for documentation, not used in computation).
        
    Returns
    -------
    connectivities : scipy.sparse.csr_matrix
        UMAP-style connectivity matrix with locally normalized probabilities.
        Values in [0,1] representing connection strengths.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist, squareform
    >>> 
    >>> # Generate sample points
    >>> points = np.random.randn(100, 2)
    >>> distances = squareform(pdist(points))
    >>> 
    >>> # Keep only k-nearest neighbors
    >>> k = 10
    >>> knn_distances = np.zeros_like(distances)
    >>> for i in range(len(distances)):
    ...     neighbors = np.argsort(distances[i])[:k+1]  # +1 for self
    ...     knn_distances[i, neighbors] = distances[i, neighbors]
    >>> 
    >>> # Convert to UMAP connectivities
    >>> conn = to_connectivities_matrix(knn_distances, k_neighbors=k)
    >>> 
    >>> # Check properties
    >>> print(f"Min: {conn.data.min():.3f}, Max: {conn.data.max():.3f}")
    >>> print(f"Density: {conn.nnz / (conn.shape[0]**2):.3f}")
    """
    # Convert to sparse format if needed
    if not sparse.issparse(distance_matrix):
        distance_matrix = sparse.csr_matrix(distance_matrix)
    else:
        distance_matrix = distance_matrix.tocsr()
    
    n_points = distance_matrix.shape[0]
    
    # Ensure diagonal is zero (no self-distances)
    distance_matrix.setdiag(0)
    distance_matrix.eliminate_zeros()
    
    # Convert distances to local probabilities
    local_probabilities = _compute_local_probabilities(
        distance_matrix, 
        k_neighbors=k_neighbors,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth
    )
    
    # Symmetrize using fuzzy union (UMAP's approach)
    connectivities = _fuzzy_union(local_probabilities)
    
    return connectivities

def to_transition_matrix(
    matrix: Union[np.ndarray, sparse.spmatrix],
    symmetrize: bool = True,
    method: str = 'max'
) -> sparse.csr_matrix:
    """
    Convert a matrix to a row-normalized transition matrix.
    
    Creates a transition probability matrix where each row sums to 1,
    representing the probability of transitioning from one node to its
    neighbors. Follows UMAP conventions for probability calculations.
    
    Parameters
    ----------
    matrix : array-like or sparse matrix
        Input matrix with connection weights/affinities.
        Higher values indicate stronger connections.
        
    symmetrize : bool, default=True
        Whether to symmetrize the matrix before normalization.
        
    method : str, default='max'
        Method for symmetrization when symmetrize=True:
        - 'max': Use maximum of (i,j) and (j,i) entries
        - 'min': Use minimum of (i,j) and (j,i) entries
        - 'mean': Use average of (i,j) and (j,i) entries
        
    Returns
    -------
    transitions : scipy.sparse.csr_matrix
        Row-normalized transition matrix where each row sums to 1
        (or 0 for isolated nodes).
        
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Create a weighted adjacency matrix
    >>> weights = np.array([[0, 0.8, 0.4], 
    ...                     [0.8, 0, 0.2],
    ...                     [0.4, 0.2, 0]])
    >>> 
    >>> # Convert to transition matrix
    >>> trans = to_transition_matrix(weights)
    >>> print(trans.toarray())
    [[0.    0.667 0.333]
     [0.8   0.    0.2  ]
     [0.667 0.333 0.   ]]
     
    >>> # Check row sums (should be 1 for connected nodes)
    >>> print(trans.sum(axis=1))
    [[1.]
     [1.]
     [1.]]
    """
    # Convert to sparse format if needed
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    
    # Ensure non-negative weights
    transitions = matrix.copy()
    transitions.data = np.maximum(transitions.data, 0)
    
    if symmetrize:
        transitions = _symmetrize_matrix(transitions, method=method)
    
    # Ensure diagonal is zero (no self-transitions)
    transitions.setdiag(0)
    transitions.eliminate_zeros()
    
    # Row-normalize to create transition probabilities
    transitions = _row_normalize_matrix(transitions)
    
    return transitions


def _compute_local_probabilities(
    distance_matrix: sparse.spmatrix,
    k_neighbors: int = 15,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
    max_iter: int = 64,
    tolerance: float = 1e-5
) -> sparse.csr_matrix:
    """
    Convert distances to local probabilities using UMAP's algorithm.
    
    For each point i, solves for σᵢ such that:
    Σⱼ exp(-(dᵢⱼ-ρᵢ)/σᵢ) ≈ log₂(k_neighbors)
    
    This creates locally normalized neighborhoods where each point
    has roughly the same "effective" number of neighbors regardless
    of local density variations.
    
    Parameters
    ----------
    distance_matrix : sparse matrix
        k-NN distance matrix with zeros for non-neighbors.
        
    k_neighbors : int
        Target number of effective neighbors (perplexity-like).
        
    local_connectivity : float
        Number of nearest neighbors with probability ≈ 1.0.
        
    bandwidth : float
        Global bandwidth multiplier.
        
    max_iter : int
        Maximum iterations for binary search.
        
    tolerance : float
        Convergence tolerance for binary search.
        
    Returns
    -------
    probabilities : sparse.csr_matrix
        Local probability matrix pᵢⱼ = exp(-(dᵢⱼ-ρᵢ)/σᵢ)
    """
    distance_matrix = distance_matrix.tocsr()
    n_points = distance_matrix.shape[0]
    
    # Target entropy (log perplexity)
    target_entropy = np.log2(k_neighbors)
    
    # Initialize probability matrix
    probabilities = distance_matrix.copy().astype(np.float32)
    
    # Process each point individually
    for i in range(n_points):
        # Get non-zero distances for point i
        start_idx = distance_matrix.indptr[i]
        end_idx = distance_matrix.indptr[i + 1]
        
        if start_idx == end_idx:  # No neighbors
            continue
            
        distances = distance_matrix.data[start_idx:end_idx].copy()
        neighbor_indices = distance_matrix.indices[start_idx:end_idx].copy()
        
        # Remove self-connections
        non_self_mask = neighbor_indices != i
        distances = distances[non_self_mask]
        neighbor_indices = neighbor_indices[non_self_mask]
        
        if len(distances) == 0:
            continue
        
        # Compute ρᵢ (distance to closest neighbor)
        rho_i = np.min(distances) if local_connectivity > 0 else 0.0
        
        # Binary search for σᵢ
        sigma_i = _find_optimal_sigma(
            distances, rho_i, target_entropy, bandwidth,
            max_iter, tolerance
        )
        
        # Convert distances to probabilities
        local_probs = np.exp(-(distances - rho_i) / (sigma_i * bandwidth))
        
        # Store back in matrix
        probabilities.data[start_idx:end_idx][non_self_mask] = local_probs
    
    # Zero out diagonal and clean up
    probabilities.setdiag(0)
    probabilities.eliminate_zeros()
    
    return probabilities


def _find_optimal_sigma(
    distances: np.ndarray,
    rho: float,
    target_entropy: float,
    bandwidth: float = 1.0,
    max_iter: int = 64,
    tolerance: float = 1e-5
) -> float:
    """
    Binary search to find σ such that entropy ≈ target_entropy.
    
    Solves: Σⱼ exp(-(dⱼ-ρ)/σ) ≈ 2^target_entropy
    
    Parameters
    ----------
    distances : array
        Distances to neighbors.
        
    rho : float
        Distance to closest neighbor.
        
    target_entropy : float
        Target log perplexity (log₂(k)).
        
    bandwidth : float
        Global bandwidth multiplier.
        
    max_iter : int
        Maximum binary search iterations.
        
    tolerance : float
        Convergence tolerance.
        
    Returns
    -------
    sigma : float
        Optimal bandwidth parameter.
    """
    # Initial bounds for binary search
    sigma_min = 1e-10
    sigma_max = 1e3
    
    # Adjust distances
    adjusted_distances = distances - rho
    adjusted_distances = np.maximum(adjusted_distances, 1e-10)  # Prevent negative
    
    for iteration in range(max_iter):
        sigma_mid = (sigma_min + sigma_max) / 2.0
        
        # Compute probabilities and entropy
        probs = np.exp(-adjusted_distances / (sigma_mid * bandwidth))
        sum_probs = np.sum(probs)
        
        if sum_probs > 1e-10:
            # Compute entropy: H = log(Σp) + (Σp*log(p))/Σp
            log_sum_probs = np.log(sum_probs)
            entropy = log_sum_probs + np.sum(probs * (-adjusted_distances / (sigma_mid * bandwidth))) / sum_probs
        else:
            entropy = -np.inf
        
        # Binary search logic
        entropy_diff = entropy - target_entropy
        
        if abs(entropy_diff) < tolerance:
            break
            
        if entropy_diff > 0:  # Too much entropy, decrease sigma
            sigma_max = sigma_mid
        else:  # Too little entropy, increase sigma
            sigma_min = sigma_mid
    
    return (sigma_min + sigma_max) / 2.0


def _fuzzy_union(probabilities: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Symmetrize probability matrix using fuzzy union.
    
    UMAP's symmetrization: wᵢⱼ = pᵢⱼ + pⱼᵢ - pᵢⱼpⱼᵢ
    
    This isn't simple averaging - it's a probabilistic OR operation.
    If either direction has high probability, the edge is strong.
    
    Parameters
    ----------
    probabilities : sparse matrix
        Asymmetric local probabilities.
        
    Returns
    -------
    symmetric_probabilities : sparse.csr_matrix
        Symmetrized probability matrix.
    """
    probabilities = probabilities.tocsr()
    probabilities_t = probabilities.T.tocsr()
    
    # Compute fuzzy union: A ∪ B = A + B - A∩B = A + B - A*B
    # For probabilities: p₁ ∪ p₂ = p₁ + p₂ - p₁*p₂
    
    # Element-wise sum
    union = probabilities + probabilities_t
    
    # Element-wise product (intersection)
    intersection = probabilities.multiply(probabilities_t)
    
    # Fuzzy union
    symmetric = union - intersection
    
    # Clip to [0,1] and ensure format
    symmetric.data = np.clip(symmetric.data, 0.0, 1.0)
    symmetric = symmetric.tocsr()
    symmetric.eliminate_zeros()
    
    return symmetric


def _symmetrize_matrix(
    matrix: sparse.spmatrix, 
    method: str = 'max'
) -> sparse.csr_matrix:
    """
    Symmetrize a sparse matrix using specified method.
    
    Parameters
    ----------
    matrix : sparse matrix
        Input matrix to symmetrize.
        
    method : str
        Symmetrization method ('max', 'min', 'mean', 'or').
        
    Returns
    -------
    symmetric_matrix : sparse.csr_matrix
        Symmetrized matrix.
    """
    matrix = matrix.tocsr()
    matrix_t = matrix.T.tocsr()
    
    if method == 'max':
        # Element-wise maximum
        symmetric = matrix.maximum(matrix_t)
    elif method == 'min':
        # Element-wise minimum (only keep edges present in both directions)
        symmetric = matrix.minimum(matrix_t)
    elif method == 'mean':
        # Element-wise average
        symmetric = (matrix + matrix_t) / 2.0
    elif method == 'or':
        # Logical OR - connection exists if either direction has edge
        binary_matrix = matrix.copy()
        binary_matrix.data = (binary_matrix.data > 0).astype(np.float32)
        binary_matrix_t = matrix_t.copy()
        binary_matrix_t.data = (binary_matrix_t.data > 0).astype(np.float32)
        symmetric = binary_matrix.maximum(binary_matrix_t)
    else:
        raise ValueError(f"Unknown symmetrization method: {method}")
    
    return symmetric


def _row_normalize_matrix(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Row-normalize a sparse matrix to create transition probabilities.
    
    Each row is normalized to sum to 1. Rows with all zeros remain zero.
    
    Parameters
    ----------
    matrix : sparse matrix
        Input matrix to normalize.
        
    Returns
    -------
    normalized_matrix : sparse.csr_matrix
        Row-normalized matrix.
    """
    matrix = matrix.tocsr()
    
    # Compute row sums
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    
    # Avoid division by zero
    nonzero_rows = row_sums > 0
    
    # Create diagonal matrix for normalization
    row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    row_sums[~nonzero_rows] = 0.0
    
    # Apply normalization
    diag_matrix = sparse.diags(row_sums, format='csr')
    normalized = diag_matrix @ matrix
    
    return normalized

def matrix_stats(matrix: Union[np.ndarray, sparse.spmatrix]) -> dict:
    """
    Compute statistics for a matrix (useful for debugging/validation).
    
    Parameters
    ----------
    matrix : array-like or sparse matrix
        Input matrix to analyze.
        
    Returns
    -------
    stats : dict
        Dictionary containing matrix statistics:
        - 'shape': Matrix dimensions
        - 'nnz': Number of non-zero elements
        - 'density': Fraction of non-zero elements
        - 'min': Minimum value
        - 'max': Maximum value
        - 'mean': Mean of non-zero values
        - 'is_symmetric': Whether matrix is symmetric
        - 'row_sums_range': (min, max) of row sums
        
    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> stats = matrix_stats(matrix)
    >>> print(stats['is_symmetric'])
    True
    >>> print(stats['density'])
    0.667
    """
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    
    # Basic statistics
    stats = {
        'shape': matrix.shape,
        'nnz': matrix.nnz,
        'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    }
    
    if matrix.nnz > 0:
        stats['min'] = matrix.data.min()
        stats['max'] = matrix.data.max()
        stats['mean'] = matrix.data.mean()
    else:
        stats['min'] = stats['max'] = stats['mean'] = 0.0
    
    # Check symmetry
    if matrix.shape[0] == matrix.shape[1]:
        try:
            diff = matrix - matrix.T
            stats['is_symmetric'] = diff.nnz == 0 or np.allclose(diff.data, 0, atol=1e-10)
        except:
            stats['is_symmetric'] = False
    else:
        stats['is_symmetric'] = False
    
    # Row sum statistics
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    stats['row_sums_range'] = (row_sums.min(), row_sums.max())
    
    return stats
