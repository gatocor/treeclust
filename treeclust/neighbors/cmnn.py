"""
Consensus Shared Neighbors class for treeclust.

This module provides the ConsensusMutualNearestNeighbors class that creates consensus 
shared neighbor connectivity matrices using bootstrapping.
"""

import numpy as np
from typing import Optional
from scipy import sparse
import scipy.sparse as sp
from tqdm import tqdm
from .mnn import MutualNearestNeighbors
from ..pipelines.pipelines import PipelineBootstrapper

class ConsensusMutualNearestNeighbors(MutualNearestNeighbors):
    """
    Consensus Mutual Nearest Neighbors class using bootstrapping.
    
    This class extends MutualNearestNeighbors to create consensus mutual neighbors matrices
    by bootstrapping the data multiple times and aggregating the results.
    
    Features:
    - Inherits all MutualNearestNeighbors functionality
    - Uses bootstrapping to create multiple samples
    - Supports multiple consensus modes:
      * 'consensus': Normalized sum of mutual neighbor connections across bootstraps
      * 'stability': Fraction of times each pair was mutually connected (stability)
      * 'distance': Average distance between mutual neighbors across bootstraps
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        flavor: str = 'auto',
        mode: str = 'consensus',
        pipeline_bootstrapper: PipelineBootstrapper = None,
        n_splits: int = 10,
        keep_bootstrap_matrices: bool = False,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the ConsensusMutualNearestNeighbors class.
        
        Parameters:
        -----------
        n_neighbors : int, default=15
            Number of neighbors to compute for each point.
            
        metric : str, default='euclidean'
            Distance metric to use.
            
        flavor : str, default='auto'
            Backend implementation to use ('auto', 'cuml', 'sklearn').
            
        mode : str, default='consensus'
            Type of consensus computation:
            - 'consensus': Normalized sum of mutual neighbor connections (default)
            - 'stability': Fraction of times each pair was mutually connected 
            - 'distance': Average distance between mutual neighbors across bootstraps
            
        pipeline_bootstrapper : PipelineBootstrapper, optional
            PipelineBootstrapper instance that will be called with split_fit_transform.
            If None, creates a default PipelineBootstrapper with 80% sample ratio.
            
        n_splits : int, default=10
            Number of bootstrap iterations to perform for consensus.
            
        keep_bootstrap_matrices : bool, default=False
            Whether to store all bootstrap matrices.
            
        verbose : bool, default=True
            Whether to show progress bar during bootstrapping iterations.
            
        random_state : int, optional
            Random seed for reproducible results. Controls the randomness of the
            bootstrap sampling and ensures consistent results across runs.
        """
        # Validate consensus mode
        valid_modes = ['consensus', 'stability', 'distance']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Available options: {valid_modes}")
        
        # Determine base mode for shared neighbors computation
        if mode == 'distance':
            base_mode = 'distance'
        else:
            base_mode = 'connectivity'
            
        # Initialize parent class (MutualNearestNeighbors uses mutual nearest neighbors)
        super().__init__(
            n_neighbors=n_neighbors,
            metric=metric,
            flavor=flavor
        )
        
        # Store consensus-specific parameters
        self.consensus_mode = mode
        self.n_splits = n_splits
        self.keep_bootstrap_matrices = keep_bootstrap_matrices
        self.verbose = verbose
        self.random_state = random_state
        
        # Set up pipeline bootstrapper
        if pipeline_bootstrapper is None:
            # Create default pipeline bootstrapper with identity transform and 80% sample ratio
            from sklearn.model_selection import ShuffleSplit
            from sklearn.preprocessing import StandardScaler
            
            self.pipeline_bootstrapper = PipelineBootstrapper(
                steps=[],  # Empty pipeline for identity-like transform
                observation_splitter=ShuffleSplit(n_splits=n_splits, train_size=0.8, random_state=random_state),
                random_state=random_state,
                cache_results=False  # Memory efficient
            )
        else:
            self.pipeline_bootstrapper = pipeline_bootstrapper
            
        # Ensure n_splits is propagated to the pipeline bootstrapper and its splitters
        self.pipeline_bootstrapper.set_n_splits(n_splits)
        
        # Storage for consensus results
        self.bootstrap_matrices_ = [] if keep_bootstrap_matrices else None
                        
    def fit(self, X: np.ndarray) -> 'ConsensusMutualNearestNeighbors':
        """
        Fit the consensus shared (mutual) neighbors model using bootstrapping.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
            
        Returns:
        --------
        self : ConsensusMutualNearestNeighbors
            Returns self for method chaining.
        """
        n_samples = X.shape[0]
        
        # Initialize consensus matrix
        self.matrix_ = sparse.csr_matrix((n_samples, n_samples))
        if self.keep_bootstrap_matrices:
            self.bootstrap_matrices_ = []
        
        # Store all bootstrap matrices and their corresponding partitions
        bootstrap_matrices = []
        used_partitions = []
        
        # For distance mode, we need to track the sum of distances
        if self.consensus_mode == 'distance':
            sum_distances = sparse.csr_matrix((n_samples, n_samples))
        
        # Fit the pipeline bootstrapper and get transformed samples
        transform_iterator = self.pipeline_bootstrapper.split_fit_transform(X)
        partitions = self.pipeline_bootstrapper.get_bootstrap_partitions()
        
        # Process each bootstrap sample with optional progress bar
        bootstrap_iter = zip(transform_iterator, partitions)
        if self.verbose:
            bootstrap_iter = tqdm(bootstrap_iter, total=min(self.n_splits, len(partitions)), 
                                desc=f"CMNN Bootstrap (n_neighbors={self.n_neighbors})")
        
        for i, (X_transformed, partition) in enumerate(bootstrap_iter):
            if i >= self.n_splits:  # Limit to requested number of bootstraps
                break
                
            # Get train indices from partition
            train_indices = partition['obs_train_idx']
            used_partitions.append(train_indices)
            
            # Create temporary MutualNearestNeighbors instance for this bootstrap
            temp_sn = MutualNearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                flavor=self.flavor
            )
            
            # Fit on bootstrap sample
            bootstrap_matrix = temp_sn.fit_transform(X_transformed)
            
            # Map bootstrap matrix back to full data space
            full_matrix = self._map_to_full_space(bootstrap_matrix, train_indices, n_samples)
            bootstrap_matrices.append(full_matrix)
            
            # Store bootstrap result if requested
            if self.keep_bootstrap_matrices:
                self.bootstrap_matrices_.append(full_matrix)
            
            # Accumulate for consensus computation
            if self.consensus_mode == 'distance':
                # For distance mode: accumulate values
                sum_distances += full_matrix
            else:
                # For consensus and stability modes: accumulate binary values
                self.matrix_ += full_matrix
        
        # Finalize matrix based on consensus mode
        if self.consensus_mode == 'consensus':
            # Normalized sum of mutual neighbor connections
            # For each existing edge, count how many times both nodes could have appeared together
            self.matrix_ = self._normalize_by_potential_occurrences(self.matrix_, used_partitions)
            
        elif self.consensus_mode == 'stability':
            # Fraction of times each pair was mutually connected
            # For each existing edge, count how many times both nodes could have appeared together
            self.matrix_ = self._normalize_by_potential_occurrences(self.matrix_, used_partitions)
            
        else:  # consensus_mode == 'distance'
            # Average distances between mutual neighbors
            # Count potential occurrences for distance normalization
            count_matrix = self._count_potential_occurrences(sum_distances, used_partitions)
            
            # Avoid division by zero
            count_matrix.data[count_matrix.data == 0] = 1
            self.matrix_ = sum_distances.multiply(sparse.csr_matrix(1.0 / count_matrix.toarray()))
        
        self.is_fitted_ = True
        return self
    
    def fit_transform(self, X):
        """Fit the model and return the consensus shared neighbors matrix."""
        self.fit(X)
        return self.matrix_
    
    def get_bootstrap_matrices(self):
        """
        Get all stored bootstrap matrices.
        
        Returns:
        --------
        matrices : list of sparse matrices or None
            List of bootstrap matrices if keep_bootstrap_matrices=True, None otherwise.
        """
        if not self.keep_bootstrap_matrices:
            raise ValueError("Bootstrap matrices were not stored. Set keep_bootstrap_matrices=True during initialization.")
        return self.bootstrap_matrices_
    
    def _map_to_full_space(self, bootstrap_matrix: sp.spmatrix, train_indices: np.ndarray, n_full_samples: int) -> sp.spmatrix:
        """
        Map a bootstrap matrix back to the full sample space.
        
        Parameters:
        -----------
        bootstrap_matrix : sp.spmatrix
            Matrix computed on bootstrap sample.
        train_indices : np.ndarray
            Indices of samples used in bootstrap.
        n_full_samples : int
            Total number of samples in original dataset.
            
        Returns:
        --------
        full_matrix : sp.spmatrix
            Matrix mapped to full sample space.
        """
        # Convert to COO format for easier manipulation
        bootstrap_coo = bootstrap_matrix.tocoo()
        
        # Map indices back to full space
        full_rows = train_indices[bootstrap_coo.row]
        full_cols = train_indices[bootstrap_coo.col]
        
        # Create full space matrix
        full_matrix = sparse.csr_matrix(
            (bootstrap_coo.data, (full_rows, full_cols)),
            shape=(n_full_samples, n_full_samples)
        )
        
        return full_matrix
    
    def _normalize_by_potential_occurrences(self, matrix: sp.spmatrix, partitions: list) -> sp.spmatrix:
        """
        Normalize a sparse matrix by counting how many times each existing edge could have appeared.
        
        Parameters:
        -----------
        matrix : sp.spmatrix
            Sparse matrix with edges to normalize.
        partitions : list
            List of arrays, each containing the indices of samples used in each bootstrap.
            
        Returns:
        --------
        normalized_matrix : sp.spmatrix
            Matrix normalized by potential occurrence counts.
        """
        if matrix.nnz == 0:
            return matrix
            
        # Convert to COO format for easier manipulation
        matrix_coo = matrix.tocoo()
        n_samples = matrix.shape[0]
        
        # Create a binary matrix for each partition indicating which samples are present
        # This allows vectorized operations instead of nested loops
        partition_masks = []
        for partition in partitions:
            mask = np.zeros(n_samples, dtype=bool)
            mask[partition] = True
            partition_masks.append(mask)
        partition_masks = np.array(partition_masks)  # Shape: (n_partitions, n_samples)
        
        # Vectorized computation: for all edges at once
        # Get presence masks for all source and target nodes
        source_masks = partition_masks[:, matrix_coo.row]  # Shape: (n_partitions, n_edges)
        target_masks = partition_masks[:, matrix_coo.col]  # Shape: (n_partitions, n_edges)
        
        # Count partitions where both source and target are present for each edge
        both_present = source_masks & target_masks  # Shape: (n_partitions, n_edges)
        edge_counts = np.sum(both_present, axis=0)  # Shape: (n_edges,)
        
        # Normalize edge values by their occurrence counts (avoid division by zero)
        normalized_data = np.where(edge_counts > 0, matrix_coo.data / edge_counts, 0.0)
        
        # Create normalized matrix
        normalized_matrix = sparse.coo_matrix(
            (normalized_data, (matrix_coo.row, matrix_coo.col)),
            shape=matrix.shape
        ).tocsr()
        
        return normalized_matrix
    
    def _count_potential_occurrences(self, matrix: sp.spmatrix, partitions: list) -> sp.spmatrix:
        """
        Count how many times each existing edge could have appeared across partitions.
        
        Parameters:
        -----------
        matrix : sp.spmatrix
            Sparse matrix with edges to count.
        partitions : list
            List of arrays, each containing the indices of samples used in each bootstrap.
            
        Returns:
        --------
        count_matrix : sp.spmatrix
            Matrix with counts of potential occurrences for each edge.
        """
        if matrix.nnz == 0:
            return sparse.csr_matrix(matrix.shape)
            
        # Convert to COO format for easier manipulation
        matrix_coo = matrix.tocoo()
        n_samples = matrix.shape[0]
        
        # Create a binary matrix for each partition indicating which samples are present
        partition_masks = []
        for partition in partitions:
            mask = np.zeros(n_samples, dtype=bool)
            mask[partition] = True
            partition_masks.append(mask)
        partition_masks = np.array(partition_masks)  # Shape: (n_partitions, n_samples)
        
        # Vectorized computation: for all edges at once
        # Get presence masks for all source and target nodes
        source_masks = partition_masks[:, matrix_coo.row]  # Shape: (n_partitions, n_edges)
        target_masks = partition_masks[:, matrix_coo.col]  # Shape: (n_partitions, n_edges)
        
        # Count partitions where both source and target are present for each edge
        both_present = source_masks & target_masks  # Shape: (n_partitions, n_edges)
        edge_counts = np.sum(both_present, axis=0)  # Shape: (n_edges,)
        
        # Create count matrix
        count_matrix = sparse.coo_matrix(
            (edge_counts, (matrix_coo.row, matrix_coo.col)),
            shape=matrix.shape
        ).tocsr()
        
        return count_matrix
        
    def __repr__(self) -> str:
        """String representation of the ConsensusMutualNearestNeighbors object."""
        is_fitted = hasattr(self, 'matrix_') and self.matrix_ is not None
        return (f"ConsensusMutualNearestNeighbors(n_neighbors={self.n_neighbors}, "
                f"metric='{self.metric}', "
                f"mode='{self.consensus_mode}', "
                f"flavor='{self.flavor}', "
                f"n_splits={self.n_splits}, "
                f"keep_bootstrap_matrices={self.keep_bootstrap_matrices}, "
                f"fitted={is_fitted})")