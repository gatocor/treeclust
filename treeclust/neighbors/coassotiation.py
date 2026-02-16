"""
Coassociation distance matrix for ensemble clustering.

This module provides functionality for creating consensus/coassociation distance matrices
from multiple clustering algorithms using bootstrapping.
"""

import numpy as np
from scipy import sparse
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from typing import List, Callable, Union, Any, Optional, Tuple, Dict
import warnings
import pandas as pd

# Import centralized availability flags
from .. import IGRAPH_AVAILABLE

# Import igraph conditionally
if IGRAPH_AVAILABLE:
    import igraph as ig

# Import PipelineBootstrapper
from ..pipelines.pipelines import PipelineBootstrapper

class CoassociationDistanceMatrix:
    """
    Coassociation distance matrix for ensemble clustering with bootstrapping.
    
    This class takes a list of clustering classes (each with fit_predict method)
    and creates a consensus distance matrix based on how often pairs of samples are 
    clustered together across different clustering runs and bootstrap samples.
    
    The distance matrix is computed as 1 - C_ij, where C_ij is the coassociation
    frequency. Values range from 0.0 (always co-clustered) to 1.0 (never co-clustered).
    This handles the case where not all samples appear in every bootstrap sample.
    
    Features:
    - Supports any clustering algorithm with fit_predict method
    - Creates distance matrix (1 - coassociation) for use in hierarchical clustering
    - Handles bootstrap sampling where not all samples appear in each run
    - Memory-efficient sparse matrix implementation
    """
    
    def __init__(
        self,
        clustering_classes: Union[Any, List[Any]],
        pipeline_bootstrapper: PipelineBootstrapper = None,
        n_splits: int = 10
    ):
        """
        Initialize the CoassociationDistanceMatrix.
        
        Parameters:
        -----------
        clustering_classes : Union[Any, List[Any]]
            Single clustering class/object or list of clustering classes/objects. 
            Each must have a fit_predict(X) method that returns cluster labels. Can be:
            - Single clustering class instance (e.g., KMeans())
            - List of clustering class instances (e.g., [KMeans(), AgglomerativeClustering()])
            - Single ParameterBootstrapper instance with fit_predict method
            - List of ParameterBootstrapper instances
            - Any object(s) with fit_predict(X) -> labels method

        pipeline_bootstrapper : PipelineBootstrapper, optional
            PipelineBootstrapper instance that will be called with split_fit_transform.
            If None, creates a default PipelineBootstrapper with 80% sample ratio.
            
        n_splits : int, default=10
            Number of bootstrap runs to perform for each clustering class.
            Total clustering runs = len(clustering_classes) * n_splits
        """
        # Convert single clustering class to list for uniform handling
        if not isinstance(clustering_classes, list):
            self.clustering_classes = [clustering_classes]
        else:
            self.clustering_classes = clustering_classes
            
        # Set up pipeline bootstrapper
        if pipeline_bootstrapper is None:
            # Create default pipeline bootstrapper with identity transform and 80% sample ratio
            from sklearn.model_selection import ShuffleSplit
            from sklearn.preprocessing import StandardScaler
            
            self.pipeline_bootstrapper = PipelineBootstrapper(
                ('identity', StandardScaler()),  # Use StandardScaler as a simple identity-like transform
                observation_splitter=ShuffleSplit(n_splits=n_splits, train_size=0.8),
                cache_results=False  # Memory efficient
            )
        else:
            self.pipeline_bootstrapper = pipeline_bootstrapper
            
        self.n_splits = n_splits
        
        # Storage for results - store distance matrix as primary result
        self.distance_matrix_ = None
        self.is_fitted_ = False
        
        # Storage for hierarchical clustering results
        self.hierarchical_results_ = None
        
        # Validate clustering classes
        self._validate_clustering_classes()
        
    def _validate_clustering_classes(self):
        """Validate that all clustering classes have the required fit_predict method."""
        for i, clf in enumerate(self.clustering_classes):
            if not hasattr(clf, 'fit_predict') or not callable(getattr(clf, 'fit_predict')):
                raise ValueError(
                    f"Clustering class at index {i} does not have a callable 'fit_predict' method. "
                    f"All clustering classes must implement fit_predict(X) -> labels."
                )
    
    def fit(self, X: np.ndarray) -> 'CoassociationDistanceMatrix':
        """
        Fit the coassociation distance matrix using the provided clustering classes.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
            
        Returns:
        --------
        self : CoassociationDistanceMatrix
            Returns self for method chaining.
        """
        n_samples = X.shape[0]
        
        # Use lists to store COO matrix data
        coassoc_rows = []
        coassoc_cols = []
        coassoc_data = []
        count_rows = []
        count_cols = []
        count_data = []
        
        total_runs = len(self.clustering_classes) * self.n_splits
        run_count = 0
        
        # Get bootstrap partitions and transform iterator
        if self.pipeline_bootstrapper is not None:
            transform_iterator = self.pipeline_bootstrapper.split_fit_transform(X)
            partitions = self.pipeline_bootstrapper.get_bootstrap_partitions()
            bootstrap_samples = list(zip(transform_iterator, partitions))
        else:
            # Single partition for the full dataset
            bootstrap_samples = [(X, {'obs_train_idx': np.arange(n_samples)})]
            
        for clf in self.clustering_classes:
            for run_i in range(self.n_splits):
                run_count += 1
                
                try:
                    # Get data for this run
                    sample_idx = run_i % len(bootstrap_samples)
                    X_train, partition = bootstrap_samples[sample_idx]
                    train_indices = partition['obs_train_idx']
                    
                    # Run clustering
                    labels = clf.fit_predict(X_train)
                    
                    # Update COO data lists
                    self._update_coo_data(
                        labels, train_indices, 
                        coassoc_rows, coassoc_cols, coassoc_data,
                        count_rows, count_cols, count_data
                    )
                    
                except Exception as e:
                    warnings.warn(f"Clustering run {run_count}/{total_runs} failed: {e}. Skipping.")
                    continue
        
        # Create sparse matrices from COO data with proper shape
        if coassoc_data:  # Only create if we have data
            coassoc_matrix = sparse.coo_matrix(
                (coassoc_data, (coassoc_rows, coassoc_cols)),
                shape=(n_samples, n_samples)
            )
            coassoc_matrix.sum_duplicates()
            coassoc_matrix = coassoc_matrix.tocsr()
        else:
            # Empty matrix if no coassociations found
            coassoc_matrix = sparse.csr_matrix((n_samples, n_samples))
        
        if count_data:  # Only create if we have data
            count_matrix = sparse.coo_matrix(
                (count_data, (count_rows, count_cols)),
                shape=(n_samples, n_samples)
            )
            count_matrix.sum_duplicates()
            count_matrix = count_matrix.tocsr()
        else:
            # Empty matrix if no counts found
            count_matrix = sparse.csr_matrix((n_samples, n_samples))
        
        # Normalize coassociation matrix by count matrix
        # Both matrices now guaranteed to be (n_samples, n_samples)
        normalized_coassoc = sparse.csr_matrix((n_samples, n_samples))
        
        if coassoc_matrix.nnz > 0 and count_matrix.nnz > 0:
            # Convert to COO for element-wise operations
            coassoc_coo = coassoc_matrix.tocoo()
            
            # Create normalized data arrays
            normalized_data = []
            normalized_rows = []
            normalized_cols = []
            
            # Normalize each coassociation entry by its count
            for k, (r, c, v) in enumerate(zip(coassoc_coo.row, coassoc_coo.col, coassoc_coo.data)):
                count_val = count_matrix[r, c]
                if count_val > 0:
                    normalized_data.append(v / count_val)
                    normalized_rows.append(r)
                    normalized_cols.append(c)
            
            # Create normalized sparse matrix
            if normalized_data:  # Only if we have normalized data
                normalized_coassoc = sparse.coo_matrix(
                    (normalized_data, (normalized_rows, normalized_cols)),
                    shape=(n_samples, n_samples)
                ).tocsr()
        
        # Convert to distance matrix: distance = 1 - coassociation
        # For pairs never evaluated together, distance = 1.0 (maximum)
        # For pairs always co-clustered, distance = 0.0 (minimum)
        
        # Start with distance matrix of all 1s (maximum distance) for pairs never seen together
        # Create dense matrix since sparse matrices with mostly 1s are inefficient
        distance_matrix = np.ones((n_samples, n_samples))
        
        if normalized_coassoc.nnz > 0:
            # Convert coassociation to distance: distance = 1 - coassociation
            # Only update positions that have coassociation data
            coassoc_dense = normalized_coassoc.toarray()
            # For positions with coassociation data, set distance = 1 - coassociation
            mask = coassoc_dense > 0  # Only update where we have coassociation data
            distance_matrix[mask] = 1.0 - coassoc_dense[mask]
        
        # Ensure diagonal is 0 (distance from sample to itself)
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Convert back to sparse matrix for storage efficiency
        self.distance_matrix_ = sparse.csr_matrix(distance_matrix)
        self.is_fitted_ = True
        
        return self
    
    def _update_coo_data(
        self, 
        labels: np.ndarray, 
        sample_indices: np.ndarray, 
        coassoc_rows: List[int],
        coassoc_cols: List[int],
        coassoc_data: List[float],
        count_rows: List[int],
        count_cols: List[int],
        count_data: List[int]
    ):
        """
        Update COO sparse matrix data based on clustering results.
        
        Parameters:
        -----------
        labels : np.ndarray
            Cluster labels for the bootstrap samples.
        sample_indices : np.ndarray
            Indices of samples in the full dataset that were used in this bootstrap.
        coassoc_rows, coassoc_cols, coassoc_data : List
            COO data for coassociation matrix.
        count_rows, count_cols, count_data : List
            COO data for count matrix.
        """
        n_bootstrap_samples = len(labels)
        
        # For each pair of samples in the bootstrap
        for i in range(n_bootstrap_samples):
            for j in range(i, n_bootstrap_samples):  # Only upper triangle + diagonal
                # Map bootstrap indices to full dataset indices
                full_i = sample_indices[i]
                full_j = sample_indices[j]
                
                # Add to count matrix data (this pair was evaluated)
                count_rows.extend([full_i, full_j])
                count_cols.extend([full_j, full_i])
                count_data.extend([1, 1])  # Symmetric
                
                # Add to coassociation data if same cluster
                if labels[i] == labels[j]:
                    coassoc_rows.extend([full_i, full_j])
                    coassoc_cols.extend([full_j, full_i])
                    coassoc_data.extend([1.0, 1.0])  # Symmetric
    
    def fit_transform(self, X: np.ndarray) -> sp.spmatrix:
        """
        Fit the coassociation distance matrix and return it.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
            
        Returns:
        --------
        distance_matrix : sp.spmatrix
            Coassociation distance matrix where entry (i,j) = 1 - coassociation(i,j).
            Values range from 0.0 (always co-clustered) to 1.0 (never co-clustered).
            Diagonal elements are always 0.0 (distance from sample to itself).
        """
        self.fit(X)
        return self.distance_matrix_
    
    def get_coassociation_matrix(self) -> sp.spmatrix:
        """
        Get the coassociation matrix derived from the distance matrix.
        
        The coassociation matrix is computed as 1 - distance_matrix.
        
        Returns:
        --------
        coassociation_matrix : sp.spmatrix
            Coassociation matrix where entry (i,j) represents the fraction
            of clustering runs where samples i and j were assigned to the same cluster.
            Values range from 0.0 (never co-clustered) to 1.0 (always co-clustered).
            
        Raises:
        -------
        ValueError
            If the matrix has not been fitted yet.
        """
        if not self.is_fitted_:
            raise ValueError("Distance matrix has not been fitted yet. Call fit() first.")
        
        # Convert distance back to coassociation: coassociation = 1 - distance
        # Since the distance matrix may be sparse, we need to handle this carefully
        if sparse.issparse(self.distance_matrix_):
            # Convert sparse distance matrix to dense for proper conversion
            distance_dense = self.distance_matrix_.toarray()
            coassoc_dense = 1.0 - distance_dense
            coassoc_matrix = sparse.csr_matrix(coassoc_dense)
        else:
            coassoc_matrix = 1.0 - self.distance_matrix_
        
        # Ensure diagonal is 1.0 (coassociation of sample with itself)
        if sparse.issparse(coassoc_matrix):
            coassoc_matrix.setdiag(1.0)
        else:
            np.fill_diagonal(coassoc_matrix, 1.0)
        
        return coassoc_matrix
    
    def get_distance_matrix(self) -> sp.spmatrix:
        """
        Get the fitted distance matrix (primary result).
        
        Returns:
        --------
        distance_matrix : sp.spmatrix
            Distance matrix where entry (i,j) = 1 - coassociation(i,j).
            Values range from 0.0 (always co-clustered) to 1.0 (never co-clustered).
            
        Raises:
        -------
        ValueError
            If the matrix has not been fitted yet.
        """
        if not self.is_fitted_:
            raise ValueError("Distance matrix has not been fitted yet. Call fit() first.")
        return self.distance_matrix_
    
    def to_dense(self) -> np.ndarray:
        """
        Convert the distance matrix to dense format.
        
        Returns:
        --------
        dense_matrix : np.ndarray
            Dense distance matrix.
        """
        if not self.is_fitted_:
            raise ValueError("Distance matrix has not been fitted yet. Call fit() first.")
        return self.distance_matrix_.toarray()
    
    def __repr__(self) -> str:
        """String representation of the CoassociationDistanceMatrix."""
        n_classes = len(self.clustering_classes)
        total_runs = n_classes * self.n_splits
        has_bootstrap = self.pipeline_bootstrapper is not None
        
        return (f"CoassociationDistanceMatrix(n_clustering_classes={n_classes}, "
                f"n_splits={self.n_splits}, "
                f"total_runs={total_runs}, "
                f"with_bootstrap={has_bootstrap}, "
                f"fitted={self.is_fitted_})")
    
    def hierarchical_clustering(self, 
                               linkage_method: str = 'complete',
                               max_hierarchy_levels: Optional[int] = None,
                               min_cluster_size: int = 2,
                               level_strategy: str = 'natural') -> Dict[str, Any]:
        """
        Perform hierarchical clustering on the distance matrix with cluster stability analysis.
        
        Parameters
        ----------
        linkage_method : str, default='complete'
            Method for computing the linkage. Options: 'ward', 'complete', 'average', 'single'
        max_hierarchy_levels : int, optional
            Maximum number of hierarchy levels to analyze. If None, analyzes all levels.
        min_cluster_size : int, default=2
            Minimum size for a cluster to be considered valid.
        level_strategy : str, default='natural'
            Strategy for creating hierarchy levels:
            - 'powers_of_2': Create levels with 2^0, 2^1, 2^2, ... clusters
            - 'natural': Use natural merge distances from dendrogram
            - 'linear': Create levels with 1, 2, 3, ... clusters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'linkage_matrix': Linkage matrix from hierarchical clustering
            - 'dendrogram_data': Data for plotting dendrogram
            - 'cluster_assignments': Dict mapping hierarchy level to cluster assignments
            - 'cluster_stability': Dict mapping hierarchy level to stability scores
            - 'hierarchy_graph': NetworkX graph representing the hierarchy with stability
        """
        if not hasattr(self, 'distance_matrix_'):
            raise ValueError("Must call fit() before hierarchical_clustering()")
            
        # Convert sparse matrix to dense for hierarchical clustering
        if sparse.issparse(self.distance_matrix_):
            dense_distance = self.distance_matrix_.toarray()
        else:
            dense_distance = self.distance_matrix_
            
        # Ensure matrix is symmetric and valid
        dense_distance = np.maximum(dense_distance, dense_distance.T)
        np.fill_diagonal(dense_distance, 0)
        
        # Convert to condensed distance matrix for linkage
        condensed_distance = squareform(dense_distance)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distance, method=linkage_method)
        
        # Generate dendrogram data
        dendrogram_data = dendrogram(linkage_matrix, no_plot=True)
        
        # Determine hierarchy levels to analyze
        n_samples = dense_distance.shape[0]
        if max_hierarchy_levels is None:
            max_hierarchy_levels = n_samples - 1
        else:
            max_hierarchy_levels = min(max_hierarchy_levels, n_samples - 1)
            
        # Analyze cluster assignments and stability at each level
        cluster_assignments = {}
        cluster_stability = {}
        
        if level_strategy == 'powers_of_2':
            # Create levels with 2^0, 2^1, 2^2, ... clusters
            print(f"Using powers of 2 strategy for hierarchy levels")
            
            if max_hierarchy_levels is None:
                # Automatically determine max levels based on data size
                max_hierarchy_levels = int(np.log2(n_samples)) + 1
            
            for level in range(max_hierarchy_levels + 1):
                target_clusters = 2**level
                
                # Don't ask for more clusters than we have samples
                if target_clusters > n_samples:
                    break
                    
                clusters = fcluster(linkage_matrix, target_clusters, criterion='maxclust')
                actual_n_clusters = len(np.unique(clusters))
                
                cluster_assignments[level] = clusters
                
                # Calculate cluster stability
                stability_scores = self._calculate_cluster_stability(clusters, min_cluster_size)
                cluster_stability[level] = stability_scores
                
                cluster_sizes = np.bincount(clusters)[1:]
                print(f"Level {level}: requested {target_clusters} clusters, got {actual_n_clusters} clusters, sizes: {cluster_sizes}")
                
                # Stop if we can't create more clusters
                if actual_n_clusters < target_clusters:
                    break
                    
        elif level_strategy == 'linear':
            # Create levels with 1, 2, 3, ... clusters
            print(f"Using linear strategy for hierarchy levels")
            
            if max_hierarchy_levels is None:
                max_hierarchy_levels = min(20, n_samples)  # Reasonable default
            
            for level in range(max_hierarchy_levels + 1):
                target_clusters = level + 1
                
                if target_clusters > n_samples:
                    break
                    
                clusters = fcluster(linkage_matrix, target_clusters, criterion='maxclust')
                actual_n_clusters = len(np.unique(clusters))
                
                cluster_assignments[level] = clusters
                
                # Calculate cluster stability
                stability_scores = self._calculate_cluster_stability(clusters, min_cluster_size)
                cluster_stability[level] = stability_scores
                
                cluster_sizes = np.bincount(clusters)[1:]
                print(f"Level {level}: requested {target_clusters} clusters, got {actual_n_clusters} clusters, sizes: {cluster_sizes}")
                
        elif level_strategy == 'natural':
            # Use natural merge distances from dendrogram (original approach)
            print(f"Using natural distance strategy for hierarchy levels")
            
            # Extract merge distances from linkage matrix to define natural levels
            merge_distances = linkage_matrix[:, 2]  # Third column contains merge distances
            
            # Find unique merge distances (these represent natural split points)
            unique_distances = np.unique(merge_distances)
            
            # Sort distances in descending order (start from most separated clusters)
            # We want to start with maximum separation and work towards finer divisions
            cut_distances = np.sort(unique_distances)[::-1]
            
            # Limit the number of levels to explore
            if max_hierarchy_levels is not None:
                cut_distances = cut_distances[:max_hierarchy_levels + 1]
            
            print(f"Natural hierarchy: using {len(cut_distances)} distance thresholds")
            
            level = 0
            previous_n_clusters = 0
            
            for distance_threshold in cut_distances:
                # Cut dendrogram at this distance
                clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
                actual_n_clusters = len(np.unique(clusters))
                
                # Skip if this doesn't create a new clustering structure
                if actual_n_clusters == previous_n_clusters and level > 0:
                    continue
                    
                # Skip if we get too many small clusters (but be less strict)
                cluster_sizes = np.bincount(clusters)[1:]  # Skip index 0
                if len(cluster_sizes) > 0 and min(cluster_sizes) < min_cluster_size and actual_n_clusters > 10:
                    continue
                
                cluster_assignments[level] = clusters
                
                # Calculate cluster stability
                stability_scores = self._calculate_cluster_stability(clusters, min_cluster_size)
                cluster_stability[level] = stability_scores
                
                print(f"Level {level}: distance threshold {distance_threshold:.4f} -> {actual_n_clusters} clusters, sizes: {cluster_sizes}")
                
                previous_n_clusters = actual_n_clusters
                level += 1
                
                # Stop if we reach single-sample clusters or too many clusters
                if actual_n_clusters >= n_samples // 2:
                    break
        
        else:
            raise ValueError(f"Unknown level_strategy: {level_strategy}. Choose from 'powers_of_2', 'linear', or 'natural'.")
            
        # Create hierarchy graph
        actual_max_level = max(cluster_assignments.keys()) if cluster_assignments else 0
        hierarchy_graph = self._create_hierarchy_graph(linkage_matrix, cluster_assignments, 
                                                     cluster_stability, actual_max_level)
        
        # Store results in instance variable
        self.hierarchical_results_ = {
            'linkage_matrix': linkage_matrix,
            'dendrogram_data': dendrogram_data,
            'cluster_assignments': cluster_assignments,
            'cluster_stability': cluster_stability,
            'hierarchy_graph': hierarchy_graph
        }
        
        return self.hierarchical_results_
    
    def _calculate_cluster_stability(self, clusters: np.ndarray, min_cluster_size: int) -> Dict[int, float]:
        """
        Calculate stability scores for clusters based on coassociation strength.
        
        Parameters
        ----------
        clusters : np.ndarray
            Cluster assignments for each sample
        min_cluster_size : int
            Minimum cluster size to consider
            
        Returns
        -------
        Dict[int, float]
            Mapping from cluster ID to stability score
        """
        if not hasattr(self, 'coassociation_matrix_'):
            # If no coassociation matrix, use distance-based stability
            return self._distance_based_stability(clusters, min_cluster_size)
            
        stability_scores = {}
        unique_clusters = np.unique(clusters)
        
        # Get coassociation matrix in dense format for calculations
        if sparse.issparse(self.coassociation_matrix_):
            coassoc_dense = self.coassociation_matrix_.toarray()
        else:
            coassoc_dense = self.coassociation_matrix_
            
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < min_cluster_size:
                stability_scores[cluster_id] = 0.0
                continue
                
            # Get indices of samples in this cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            # Calculate internal coassociation strength
            if len(cluster_indices) > 1:
                # Average coassociation within cluster
                internal_coassoc = coassoc_dense[np.ix_(cluster_indices, cluster_indices)]
                # Exclude diagonal
                mask = ~np.eye(len(cluster_indices), dtype=bool)
                internal_strength = np.mean(internal_coassoc[mask])
                
                # Calculate external coassociation (with other clusters)
                external_indices = np.where(~cluster_mask)[0]
                if len(external_indices) > 0:
                    external_coassoc = coassoc_dense[np.ix_(cluster_indices, external_indices)]
                    external_strength = np.mean(external_coassoc)
                else:
                    external_strength = 0.0
                
                # Stability is ratio of internal to external coassociation
                if external_strength > 0:
                    stability = internal_strength / (internal_strength + external_strength)
                else:
                    stability = internal_strength
            else:
                stability = 1.0  # Single sample cluster is perfectly stable
                
            stability_scores[cluster_id] = float(stability)
            
        return stability_scores
    
    def _distance_based_stability(self, clusters: np.ndarray, min_cluster_size: int) -> Dict[int, float]:
        """
        Calculate stability based on distance matrix when coassociation matrix is not available.
        
        Parameters
        ----------
        clusters : np.ndarray
            Cluster assignments for each sample
        min_cluster_size : int
            Minimum cluster size to consider
            
        Returns
        -------
        Dict[int, float]
            Mapping from cluster ID to stability score
        """
        stability_scores = {}
        unique_clusters = np.unique(clusters)
        
        # Get distance matrix in dense format
        if sparse.issparse(self.distance_matrix_):
            dist_dense = self.distance_matrix_.toarray()
        else:
            dist_dense = self.distance_matrix_
            
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < min_cluster_size:
                stability_scores[cluster_id] = 0.0
                continue
                
            # Get indices of samples in this cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 1:
                # Calculate internal distances (smaller is better)
                internal_dist = dist_dense[np.ix_(cluster_indices, cluster_indices)]
                mask = ~np.eye(len(cluster_indices), dtype=bool)
                avg_internal_dist = np.mean(internal_dist[mask])
                
                # Calculate external distances
                external_indices = np.where(~cluster_mask)[0]
                if len(external_indices) > 0:
                    external_dist = dist_dense[np.ix_(cluster_indices, external_indices)]
                    avg_external_dist = np.mean(external_dist)
                else:
                    avg_external_dist = 1.0
                
                # Stability based on silhouette-like measure
                if avg_external_dist > avg_internal_dist:
                    stability = (avg_external_dist - avg_internal_dist) / max(avg_external_dist, avg_internal_dist)
                else:
                    stability = 0.0
            else:
                stability = 1.0
                
            stability_scores[cluster_id] = float(max(0.0, stability))
            
        return stability_scores
    
    def _create_hierarchy_graph(self, linkage_matrix: np.ndarray, 
                              cluster_assignments: Dict[int, np.ndarray],
                              cluster_stability: Dict[int, Dict[int, float]],
                              max_levels: int) -> Any:
        """
        Create an igraph.Graph representing the clustering hierarchy.
        
        Parameters
        ----------
        linkage_matrix : np.ndarray
            Linkage matrix from hierarchical clustering
        cluster_assignments : Dict[int, np.ndarray]
            Cluster assignments for each hierarchy level
        cluster_stability : Dict[int, Dict[int, float]]
            Stability scores for each cluster at each level
        max_levels : int
            Maximum hierarchy levels
            
        Returns
        -------
        igraph.Graph or dict
            If igraph is available, returns igraph.Graph with stability as node attributes 
            and membership probabilities as edge weights.
            If igraph is not available, returns dictionary representation.
        """
        if not IGRAPH_AVAILABLE:
            # Return dictionary representation if igraph is not available
            warnings.warn("igraph not available. Returning dictionary representation of hierarchy.")
            
            hierarchy_dict = {
                'nodes': [],
                'edges': [],
                'levels': cluster_assignments,
                'stability': cluster_stability
            }
            
            # Add nodes
            node_id = 0
            level_to_node_ids = {}
            
            for level in sorted(cluster_assignments.keys()):
                clusters = cluster_assignments[level]
                stability = cluster_stability[level]
                level_nodes = {}
                
                for cluster_id in np.unique(clusters):
                    cluster_size = np.sum(clusters == cluster_id)
                    cluster_members = np.where(clusters == cluster_id)[0].tolist()
                    
                    node_data = {
                        'id': node_id,
                        'level': level,
                        'cluster_id': cluster_id,
                        'stability': stability.get(cluster_id, 0.0),
                        'size': cluster_size,
                        'members': cluster_members
                    }
                    hierarchy_dict['nodes'].append(node_data)
                    level_nodes[cluster_id] = node_id
                    node_id += 1
                    
                level_to_node_ids[level] = level_nodes
            
            # Add edges
            sorted_levels = sorted(cluster_assignments.keys())
            for i in range(len(sorted_levels) - 1):
                current_level = sorted_levels[i]
                next_level = sorted_levels[i + 1]
                
                current_clusters = cluster_assignments[current_level]
                next_clusters = cluster_assignments[next_level]
                
                current_nodes = level_to_node_ids[current_level]
                next_nodes = level_to_node_ids[next_level]
                
                for curr_cluster_id, curr_node_id in current_nodes.items():
                    curr_members = set(np.where(current_clusters == curr_cluster_id)[0])
                    
                    for next_cluster_id, next_node_id in next_nodes.items():
                        next_members = set(np.where(next_clusters == next_cluster_id)[0])
                        
                        overlap = len(curr_members.intersection(next_members))
                        if overlap > 0:
                            weight = overlap / len(curr_members)
                            edge_data = {
                                'source': curr_node_id,
                                'target': next_node_id,
                                'weight': weight,
                                'overlap': overlap
                            }
                            hierarchy_dict['edges'].append(edge_data)
            
            return hierarchy_dict
        
        # Create igraph representation
        vertices = []
        edges = []
        node_id = 0
        level_to_node_ids = {}
        
        # Add vertices for each cluster at each level
        for level in sorted(cluster_assignments.keys()):
            clusters = cluster_assignments[level]
            stability = cluster_stability[level]
            level_nodes = {}
            
            for cluster_id in np.unique(clusters):
                cluster_size = np.sum(clusters == cluster_id)
                cluster_members = np.where(clusters == cluster_id)[0].tolist()
                
                vertices.append({
                    'name': f"L{level}_C{cluster_id}",
                    'id': node_id,
                    'level': level,
                    'cluster_id': cluster_id,
                    'stability': stability.get(cluster_id, 0.0),
                    'size': cluster_size,
                    'members': cluster_members
                })
                
                level_nodes[cluster_id] = node_id
                node_id += 1
                
            level_to_node_ids[level] = level_nodes
        
        # Calculate edges between consecutive levels
        edge_list = []
        edge_weights = []
        edge_overlaps = []
        
        sorted_levels = sorted(cluster_assignments.keys())
        for i in range(len(sorted_levels) - 1):
            current_level = sorted_levels[i]
            next_level = sorted_levels[i + 1]
            
            current_clusters = cluster_assignments[current_level]
            next_clusters = cluster_assignments[next_level]
            
            current_nodes = level_to_node_ids[current_level]
            next_nodes = level_to_node_ids[next_level]
            
            # Calculate membership overlap between levels
            for curr_cluster_id, curr_node_id in current_nodes.items():
                curr_members = set(np.where(current_clusters == curr_cluster_id)[0])
                
                for next_cluster_id, next_node_id in next_nodes.items():
                    next_members = set(np.where(next_clusters == next_cluster_id)[0])
                    
                    # Calculate overlap
                    overlap = len(curr_members.intersection(next_members))
                    if overlap > 0:
                        # Edge weight is proportion of current cluster that goes to next cluster
                        weight = overlap / len(curr_members)
                        edge_list.append((curr_node_id, next_node_id))
                        edge_weights.append(weight)
                        edge_overlaps.append(overlap)
        
        # Create igraph Graph
        g = ig.Graph(directed=True)
        g.add_vertices(len(vertices))
        
        # Set vertex attributes
        for attr in ['name', 'id', 'level', 'cluster_id', 'stability', 'size', 'members']:
            g.vs[attr] = [v[attr] for v in vertices]
        
        # Add edges with attributes
        g.add_edges(edge_list)
        g.es['weight'] = edge_weights
        g.es['overlap'] = edge_overlaps
        
        return g
    
    # def plot_hierarchy_graph_matplotlib(self, hierarchical_result, ax, title="Hierarchy Graph"):
    #     """
    #     Plot hierarchy graph using matplotlib on a given axis.
        
    #     Parameters
    #     ----------
    #     hierarchical_result : dict
    #         Result from hierarchical_clustering() containing hierarchy_graph
    #     ax : matplotlib.axes.Axes
    #         The axis to plot on
    #     title : str
    #         Plot title
    #     """
    #     import matplotlib.pyplot as plt
        
    #     graph = hierarchical_result['hierarchy_graph']
        
    #     if not hasattr(graph, 'vs'):
    #         # Handle dictionary representation
    #         if isinstance(graph, dict):
    #             ax.text(0.5, 0.5, "igraph not available\nShowing text representation", 
    #                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
    #             ax.set_title(title)
    #             return
    #         else:
    #             ax.text(0.5, 0.5, f"Unknown graph type: {type(graph)}", 
    #                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
    #             ax.set_title(title)
    #             return
        
    #     # Set up layout
    #     try:
    #         layout = graph.layout("sugiyama")  # Hierarchical layout
    #     except:
    #         try:
    #             layout = graph.layout("fruchterman_reingold")
    #         except:
    #             layout = graph.layout("kamada_kawai")
        
    #     # Extract positions and normalize to [0, 1]
    #     positions = np.array(layout.coords)
    #     if len(positions) > 0:
    #         positions = (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0) + 1e-8)
        
    #     # Color nodes by stability score
    #     stabilities = graph.vs['stability']
    #     max_stability = max(stabilities) if stabilities else 1.0
    #     min_stability = min(stabilities) if stabilities else 0.0
        
    #     # Normalize stability scores for coloring
    #     if max_stability > min_stability:
    #         normalized_stabilities = [(s - min_stability) / (max_stability - min_stability) 
    #                                  for s in stabilities]
    #     else:
    #         normalized_stabilities = [0.5] * len(stabilities)
        
    #     # Plot edges first (so they appear behind nodes)
    #     for edge in graph.es:
    #         start_pos = positions[edge.source]
    #         end_pos = positions[edge.target]
            
    #         # Draw edge
    #         ax.annotate('', xy=end_pos, xytext=start_pos,
    #                    arrowprops=dict(arrowstyle='->', 
    #                                  alpha=0.6, 
    #                                  lw=edge['weight'] * 3,
    #                                  color='gray'))
        
    #     # Plot nodes
    #     sizes = graph.vs['size']
    #     max_size = max(sizes) if sizes else 1
    #     min_size = min(sizes) if sizes else 1
        
    #     for i, vertex in enumerate(graph.vs):
    #         pos = positions[i]
    #         stability = normalized_stabilities[i]
    #         size = vertex['size']
            
    #         # Color by stability
    #         color = plt.cm.viridis(stability)
            
    #         # Size by cluster size - make more proportional to number of cells
    #         size_range = max_size - min_size + 1
    #         node_size = 30 + (size - min_size) * (400 - 30) / size_range
            
    #         ax.scatter(pos[0], pos[1], s=node_size, c=[color], 
    #                   alpha=0.8, edgecolors='black', linewidth=1)
            
    #         # Add label
    #         ax.text(pos[0], pos[1], f"L{vertex['level']}_C{vertex['cluster_id']}", 
    #                ha='center', va='center', fontsize=8, weight='bold')
        
    #     ax.set_xlim(-0.1, 1.1)
    #     ax.set_ylim(-0.1, 1.1)
    #     ax.set_title(title, fontsize=12, weight='bold')
    #     ax.set_xlabel('Position X')
    #     ax.set_ylabel('Position Y')
    #     ax.grid(True, alpha=0.3)
    
    def plot_hierarchy_stacked_bars(self, leaf_order, ax, title="Hierarchy Stacked Bars"):
        """
        Plot hierarchy as stacked bars showing cluster divisions at each level.
        
        Parameters
        ----------
        leaf_order : array-like
            The order of leaves from dendrogram
        ax : matplotlib.axes.Axes
            The axis to plot on
        title : str
            Plot title
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        graph = self.hierarchical_results_['hierarchy_graph']
        cluster_assignments = self.hierarchical_results_['cluster_assignments']
        
        if not cluster_assignments:
            ax.text(0.5, 0.5, "No cluster assignments available", ha='center', va='center')
            return
        
        matrix_size = len(leaf_order)
        original_to_sorted = {orig_idx: sorted_idx for sorted_idx, orig_idx in enumerate(leaf_order)}
        
        # Get stability scores from graph
        stability_scores = {}
        if hasattr(graph, 'vs'):
            # igraph case
            for vertex in graph.vs:
                level = vertex['level']
                cluster_id = vertex['cluster_id']
                stability = vertex['stability']
                if level not in stability_scores:
                    stability_scores[level] = {}
                stability_scores[level][cluster_id] = stability
        elif isinstance(graph, dict) and 'nodes' in graph:
            # dict case
            for node in graph['nodes']:
                level = node['level']
                cluster_id = node['cluster_id']
                stability = node['stability']
                if level not in stability_scores:
                    stability_scores[level] = {}
                stability_scores[level][cluster_id] = stability
        
        # Sort levels from finest to coarsest (left to right)
        sorted_levels = sorted(cluster_assignments.keys(), reverse=True)  # Reverse for left to right
        n_levels = len(sorted_levels)
        
        # Plot each level as a vertical bar
        bar_width = 0.8 / n_levels  # Width of each bar
        
        for level_idx, level in enumerate(sorted_levels):
            cluster_labels = cluster_assignments[level]
            unique_clusters = np.unique(cluster_labels)
            
            # X position for this level (from left to right)
            x_pos = (level_idx + 0.5) * (1.0 / n_levels)
            
            # Calculate cluster segments
            cluster_segments = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_sample_indices = np.where(cluster_mask)[0]
                
                # Map to sorted positions
                sorted_positions = [original_to_sorted.get(idx) for idx in cluster_sample_indices if idx in original_to_sorted]
                
                if sorted_positions:
                    # Get stability score for this cluster
                    stability = stability_scores.get(level, {}).get(cluster_id, 0.0)
                    
                    cluster_segments.append({
                        'cluster_id': cluster_id,
                        'positions': sorted(sorted_positions),
                        'size': len(sorted_positions),
                        'stability': stability
                    })
            
            # Sort segments by their starting position
            cluster_segments.sort(key=lambda x: min(x['positions']))
            
            # Draw segments as rectangles
            for segment in cluster_segments:
                positions = segment['positions']
                stability = segment['stability']
                
                if len(positions) > 0:
                    # Find contiguous blocks
                    blocks = []
                    current_block = [positions[0]]
                    
                    for i in range(1, len(positions)):
                        if positions[i] == positions[i-1] + 1:
                            current_block.append(positions[i])
                        else:
                            blocks.append(current_block)
                            current_block = [positions[i]]
                    blocks.append(current_block)
                    
                    # Draw each block
                    for block in blocks:
                        # Y coordinates (inverted to match matrix orientation)
                        start_pos = 1.0 - ((max(block) + 1) / matrix_size)
                        end_pos = 1.0 - (min(block) / matrix_size)
                        height = end_pos - start_pos
                        
                        # Color based on stability (using viridis colormap)
                        color = plt.cm.viridis(stability)
                        
                        # Create rectangle
                        rect = patches.Rectangle((x_pos - bar_width/2, start_pos), 
                                               bar_width, height,
                                               facecolor=color,
                                               edgecolor='black',
                                               linewidth=0.5,
                                               alpha=0.8)
                        ax.add_patch(rect)
                        
                        # Add cluster label if block is tall enough
                        if height > 0.05:  # Only label if segment is tall enough
                            text_y = start_pos + height/2
                            ax.text(x_pos, text_y, f"C{segment['cluster_id']}", 
                                   ha='center', va='center', fontsize=6, weight='bold',
                                   color='white' if stability < 0.5 else 'black')
            
            # Add level label at the bottom
            ax.text(x_pos, -0.05, f"L{level}", ha='center', va='top', 
                   fontsize=10, weight='bold')
        
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')  # Remove all axes, ticks, and labels
        
    def plot_combined_analysis_stacked(self, true_labels=None, figsize=(16, 8)):
        """
        Create a combined plot with sorted matrix and stacked bar hierarchy visualization.
        
        Parameters
        ----------
        true_labels : array-like, optional
            True cluster labels for comparison
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        if not self.is_fitted_:
            raise ValueError("Must fit the coassociation matrix before plotting. Call fit() first.")
        
        coassoc_matrix = self.get_coassociation_matrix().toarray()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 2]})
        
        # Create a distance matrix from the coassociation matrix for proper ordering
        distance_matrix = 1 - coassoc_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Create linkage matrix directly from the coassociation distance matrix
        condensed_distances = squareform(distance_matrix, checks=False)
        sample_linkage = linkage(condensed_distances, method='complete')
        
        # Get the leaf ordering from the sample-level dendrogram
        dendro = dendrogram(sample_linkage, no_plot=True)
        leaf_order = dendro['leaves']
        
        # Sort the coassociation matrix according to the dendrogram ordering
        sorted_matrix = coassoc_matrix[np.ix_(leaf_order, leaf_order)]
        
        # Plot sorted coassociation matrix
        im = ax1.imshow(sorted_matrix, cmap='viridis', aspect='auto', origin='upper')
        ax1.set_title('Sorted Coassociation Matrix', fontsize=14, weight='bold')
        ax1.set_xlabel('Sample Index (Hierarchically Ordered)')
        ax1.set_ylabel('Sample Index (Hierarchically Ordered)')
        
        # Plot hierarchy as stacked bars
        self.plot_hierarchy_stacked_bars(leaf_order, ax2, "Clustering Hierarchy")
        
        plt.tight_layout()
        return fig
    
    def plot_hierarchy_graph_aligned(self, leaf_order, ax, title="Hierarchy Graph"):
        """
        Plot hierarchy graph with vertical layout aligned to matrix positions.
        
        Parameters
        ----------
        leaf_order : array-like
            The order of leaves from dendrogram
        ax : matplotlib.axes.Axes
            The axis to plot on
        title : str
            Plot title
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
        
        import matplotlib.pyplot as plt
        
        graph = self.hierarchical_results_['hierarchy_graph']
        cluster_assignments = self.hierarchical_results_['cluster_assignments']
        
        if isinstance(graph, dict):
            # Handle dictionary representation
            nodes = graph['nodes']
            edges = graph['edges']
            
            # Calculate node positions aligned with matrix
            node_positions = {}
            matrix_size = len(leaf_order)
            
            # Create mapping from original sample indices to sorted positions
            original_to_sorted = {orig_idx: sorted_idx for sorted_idx, orig_idx in enumerate(leaf_order)}
            
            for node in nodes:
                level = node['level']
                cluster_id = node['cluster_id']
                
                # Get the samples belonging to this cluster at this level
                if level in cluster_assignments:
                    cluster_labels = cluster_assignments[level]
                    cluster_mask = cluster_labels == cluster_id
                    cluster_sample_indices = np.where(cluster_mask)[0]
                    
                    # Map to sorted positions and calculate mean y-position
                    sorted_positions = [original_to_sorted.get(idx) for idx in cluster_sample_indices if idx in original_to_sorted]
                    
                    if sorted_positions:
                        # Y position is mean of cluster members in sorted matrix (normalized to [0,1])
                        # Invert Y to match heatmap orientation (matrix has (0,0) at top-left)
                        mean_y = 1.0 - (np.mean(sorted_positions) / matrix_size)
                        # X position based on hierarchy level (vertical layout)
                        x = level / max(n['level'] for n in nodes) if nodes else 0.5
                    else:
                        mean_y = 0.5
                        x = level / max(n['level'] for n in nodes) if nodes else 0.5
                else:
                    mean_y = 0.5
                    x = level / max(n['level'] for n in nodes) if nodes else 0.5
                
                node_positions[node['id']] = (x, mean_y)
            
            # Plot edges
            for edge in edges:
                start_pos = node_positions[edge['source']]
                end_pos = node_positions[edge['target']]
                
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       'gray', alpha=0.6, linewidth=edge['weight'] * 3)
            
            # Plot nodes
            for node in nodes:
                pos = node_positions[node['id']]
                stability = node['stability']
                size = node['size']
                
                # Color by stability
                color = plt.cm.viridis(stability)
                
                # Size by cluster size - make more proportional to number of cells
                max_size = max(n['size'] for n in nodes) if nodes else 1
                min_size = min(n['size'] for n in nodes) if nodes else 1
                # Scale from 30 to 400 pixels based on cluster size
                size_range = max_size - min_size + 1
                node_size = 30 + (size - min_size) * (400 - 30) / size_range
                
                ax.scatter(pos[0], pos[1], s=node_size, c=[color], 
                          alpha=0.8, edgecolors='black', linewidth=1)
                
                # Add label to all nodes
                ax.text(pos[0] + 0.02, pos[1], f"L{node['level']}_C{node['cluster_id']}", 
                       ha='left', va='center', fontsize=8, weight='bold')
        
        else:
            # Handle igraph.Graph
            if not hasattr(graph, 'vs'):
                ax.text(0.5, 0.5, "No valid graph data", ha='center', va='center')
                return
            
            # Calculate node positions aligned with matrix
            matrix_size = len(leaf_order)
            original_to_sorted = {orig_idx: sorted_idx for sorted_idx, orig_idx in enumerate(leaf_order)}
            
            node_positions = []
            for i, vertex in enumerate(graph.vs):
                level = vertex['level']
                cluster_id = vertex['cluster_id']
                
                # Get the samples belonging to this cluster at this level
                if level in cluster_assignments:
                    cluster_labels = cluster_assignments[level]
                    cluster_mask = cluster_labels == cluster_id
                    cluster_sample_indices = np.where(cluster_mask)[0]
                    
                    # Map to sorted positions and calculate mean y-position
                    sorted_positions = [original_to_sorted.get(idx) for idx in cluster_sample_indices if idx in original_to_sorted]
                    
                    if sorted_positions:
                        # Y position is mean of cluster members in sorted matrix (normalized to [0,1])
                        # Invert Y to match heatmap orientation
                        mean_y = 1.0 - (np.mean(sorted_positions) / matrix_size)
                    else:
                        mean_y = 0.5
                else:
                    mean_y = 0.5
                
                # X position based on hierarchy level (vertical layout)
                max_level = max(graph.vs['level'])
                x = (max_level - level) / max_level if max_level > 0 else 0.5
                
                node_positions.append((x, mean_y))
            
            # Plot edges
            for edge in graph.es:
                start_pos = node_positions[edge.source]
                end_pos = node_positions[edge.target]
                
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       'gray', alpha=0.6, linewidth=edge['weight'] * 3)
            
            # Plot nodes
            stabilities = graph.vs['stability']
            sizes = graph.vs['size']
            max_size = max(sizes) if sizes else 1
            min_size = min(sizes) if sizes else 1
            
            for i, vertex in enumerate(graph.vs):
                pos = node_positions[i]
                stability = vertex['stability']
                size = vertex['size']
                
                # Color by stability
                color = plt.cm.viridis(stability)
                
                # Size by cluster size
                size_range = max_size - min_size + 1
                node_size = 30 + (size - min_size) * (400 - 30) / size_range
                
                ax.scatter(pos[0], pos[1], s=node_size, c=[color], 
                          alpha=0.8, edgecolors='black', linewidth=1)
                
                # Add label
                ax.text(pos[0] + 0.02, pos[1], f"L{vertex['level']}_C{vertex['cluster_id']}", 
                       ha='left', va='center', fontsize=8, weight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')  # Remove all axes, ticks, and labels
    
    def plot_combined_analysis_aligned(self, true_labels=None, figsize=(16, 8)):
        """
        Create a combined plot with sorted matrix and aligned vertical hierarchy graph.
        
        Parameters
        ----------
        true_labels : array-like, optional
            True cluster labels for comparison
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        if not self.is_fitted_:
            raise ValueError("Must fit the coassociation matrix before plotting. Call fit() first.")
        
        coassoc_matrix = self.get_coassociation_matrix().toarray()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 2]})
        
        # Create a distance matrix from the coassociation matrix for proper ordering
        # Convert coassociation (similarity) to distance: distance = 1 - coassociation
        distance_matrix = 1 - coassoc_matrix
        
        # Ensure diagonal is zero (distance from sample to itself)
        np.fill_diagonal(distance_matrix, 0)
        
        # Create linkage matrix directly from the coassociation distance matrix
        # This will give us the proper sample ordering
        condensed_distances = squareform(distance_matrix, checks=False)
        sample_linkage = linkage(condensed_distances, method='complete')
        
        # Get the leaf ordering from the sample-level dendrogram
        dendro = dendrogram(sample_linkage, no_plot=True)
        leaf_order = dendro['leaves']
        
        # Sort the coassociation matrix according to the dendrogram ordering
        sorted_matrix = coassoc_matrix[np.ix_(leaf_order, leaf_order)]
        
        # Plot sorted coassociation matrix
        im = ax1.imshow(sorted_matrix, cmap='viridis', aspect='auto', origin='upper')
        ax1.set_title('Sorted Coassociation Matrix', fontsize=14, weight='bold')
        ax1.set_xlabel('Sample Index (Hierarchically Ordered)')
        ax1.set_ylabel('Sample Index (Hierarchically Ordered)')
        
        # Plot hierarchy graph with aligned positions
        self.plot_hierarchy_graph_aligned(leaf_order, ax2, "Clustering Hierarchy")
        
        # Add colorbar for stability scores on the right of the hierarchy plot
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax2)
        cbar2.set_label('Stability Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def get_cluster_labels(self, level: int) -> np.ndarray:
        """
        Get cluster labels for all cells at a given hierarchy level.
        
        Parameters
        ----------
        level : int
            Hierarchy level to get labels for
            
        Returns
        -------
        np.ndarray
            Array of cluster labels in original X matrix order, shape (n_samples,)
            
        Raises
        ------
        ValueError
            If the specified level is not found in the hierarchical result, or if
            hierarchical clustering hasn't been performed yet
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
            
        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        
        if level not in cluster_assignments:
            available_levels = sorted(cluster_assignments.keys())
            raise ValueError(f"Level {level} not found. Available levels: {available_levels}")
        
        return cluster_assignments[level].copy()
    
    def get_hierarchy_levels(self, order: str = 'top_to_bottom') -> List[int]:
        """
        Get hierarchy levels ordered from top to bottom (or bottom to top) of the tree.
        
        Parameters
        ----------
        order : str, default='top_to_bottom'
            Order of levels to return:
            - 'top_to_bottom': From coarsest clustering (fewest clusters) to finest (most clusters)
            - 'bottom_to_top': From finest clustering (most clusters) to coarsest (fewest clusters)
            - 'ascending': Same as 'top_to_bottom' (numerically ascending level numbers)
            - 'descending': Same as 'bottom_to_top' (numerically descending level numbers)
            
        Returns
        -------
        List[int]
            List of hierarchy level numbers in the requested order
            
        Raises
        ------
        ValueError
            If hierarchical clustering hasn't been performed yet or if order is invalid
            
        Examples
        --------
        >>> distance_mat = CoassociationDistanceMatrix(clusterer, n_splits=10)
        >>> distance_mat.fit(X)
        >>> distance_mat.hierarchical_clustering()
        >>> 
        >>> # Get levels from top (coarsest) to bottom (finest) of tree
        >>> levels = distance_mat.get_hierarchy_levels('top_to_bottom')
        >>> print(f"Levels from top to bottom: {levels}")
        >>> # Output: [0, 1, 2, 3, 4] (assuming 5 levels)
        >>> 
        >>> # Get levels from bottom (finest) to top (coarsest) of tree  
        >>> levels = distance_mat.get_hierarchy_levels('bottom_to_top')
        >>> print(f"Levels from bottom to top: {levels}")
        >>> # Output: [4, 3, 2, 1, 0] (assuming 5 levels)
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
            
        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        
        if not cluster_assignments:
            return []
        
        # Get all available levels
        all_levels = sorted(cluster_assignments.keys())
        
        # Return levels in the requested order
        if order in ['top_to_bottom', 'ascending']:
            return all_levels
        elif order in ['bottom_to_top', 'descending']:
            return list(reversed(all_levels))
        else:
            valid_orders = ['top_to_bottom', 'bottom_to_top', 'ascending', 'descending']
            raise ValueError(f"Invalid order '{order}'. Valid options: {valid_orders}")

    def get_cluster_labels_probabilities(self, level: int) -> np.ndarray:
        """
        Get soft cluster assignment probabilities for all cells at a given hierarchy level.
        
        This function returns a probability matrix where each cell has probabilities for 
        belonging to each cluster at the specified level. Probabilities are calculated 
        based on the coassociation matrix - higher coassociation with cluster members 
        means higher probability of belonging to that cluster.
        
        Parameters
        ----------
        level : int
            Hierarchy level to get probability assignments for
            
        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_clusters) where:
            - Each row represents a sample/cell
            - Each column represents a cluster at the specified level
            - Values are probabilities (sum to 1 for each row)
            - Entry (i, j) is the probability that sample i belongs to cluster j
            
        Raises
        ------
        ValueError
            If the specified level is not found in the hierarchical result, or if
            hierarchical clustering hasn't been performed yet
            
        Examples
        --------
        >>> distance_mat = CoassociationDistanceMatrix(clusterer, n_splits=10)
        >>> distance_mat.fit(X)
        >>> distance_mat.hierarchical_clustering()
        >>> 
        >>> # Get hard cluster labels at level 2
        >>> hard_labels = distance_mat.get_cluster_labels(level=2)
        >>> print(f"Hard labels: {hard_labels[:10]}")  # First 10 samples
        >>> 
        >>> # Get soft probability assignments at level 2
        >>> soft_probs = distance_mat.get_cluster_labels_probabilities(level=2)
        >>> print(f"Soft probabilities shape: {soft_probs.shape}")
        >>> print(f"Sample 0 probabilities: {soft_probs[0]}")  # Probabilities for sample 0
        >>> print(f"Most likely cluster for sample 0: {np.argmax(soft_probs[0])}")
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
            
        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        
        if level not in cluster_assignments:
            available_levels = sorted(cluster_assignments.keys())
            raise ValueError(f"Level {level} not found. Available levels: {available_levels}")
        
        # Get hard cluster assignments for this level
        hard_labels = cluster_assignments[level]
        unique_clusters = np.unique(hard_labels)
        n_samples = len(hard_labels)
        n_clusters = len(unique_clusters)
        
        # Get coassociation matrix
        coassoc_matrix = self.get_coassociation_matrix()
        if sparse.issparse(coassoc_matrix):
            coassoc_matrix = coassoc_matrix.toarray()
        
        # Initialize probability matrix
        prob_matrix = np.zeros((n_samples, n_clusters))
        
        # Calculate probabilities for each cluster
        for cluster_idx, cluster_id in enumerate(unique_clusters):
            # Get indices of samples currently assigned to this cluster
            cluster_member_indices = np.where(hard_labels == cluster_id)[0]
            
            if len(cluster_member_indices) == 0:
                continue
            
            # For each sample, calculate probability of belonging to this cluster
            for sample_idx in range(n_samples):
                if sample_idx in cluster_member_indices:
                    # If sample is already in this cluster, calculate mean coassociation 
                    # with other cluster members
                    other_members = cluster_member_indices[cluster_member_indices != sample_idx]
                    if len(other_members) > 0:
                        mean_coassoc = np.mean(coassoc_matrix[sample_idx, other_members])
                    else:
                        # Single-member cluster: perfect assignment
                        mean_coassoc = 1.0
                else:
                    # Sample not in this cluster: mean coassociation with all cluster members
                    mean_coassoc = np.mean(coassoc_matrix[sample_idx, cluster_member_indices])
                
                # Probability is based on coassociation strength
                # Higher coassociation = higher probability
                prob_matrix[sample_idx, cluster_idx] = mean_coassoc
        
        # Normalize probabilities so each row sums to 1
        row_sums = np.sum(prob_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        prob_matrix = prob_matrix / row_sums
        
        return prob_matrix

    def get_cluster_cell_indices(self, level: int, cluster_id: int) -> np.ndarray:
        """
        Get cell indices for a specific cluster at a given hierarchy level.
        
        Parameters
        ----------
        level : int
            Hierarchy level 
        cluster_id : int
            Cluster ID within that level
            
        Returns
        -------
        np.ndarray
            Array of cell indices (positions in original X matrix) that belong to the specified cluster
            
        Raises
        ------
        ValueError
            If the specified level or cluster_id is not found, or if hierarchical clustering 
            hasn't been performed yet
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
            
        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        
        if level not in cluster_assignments:
            available_levels = sorted(cluster_assignments.keys())
            raise ValueError(f"Level {level} not found. Available levels: {available_levels}")
        
        labels = cluster_assignments[level]
        available_clusters = np.unique(labels)
        
        if cluster_id not in available_clusters:
            raise ValueError(f"Cluster {cluster_id} not found at level {level}. "
                           f"Available clusters: {sorted(available_clusters)}")
        
        # Get indices where the cluster label matches
        cell_indices = np.where(labels == cluster_id)[0]
        return cell_indices
    
    def get_level_clusters(self, level: int) -> List[Tuple[int, int]]:
        """
        Get all cluster specifications for a given hierarchy level.
        
        Returns a list of (level, cluster_id) tuples for all clusters at the specified level.
        This format is compatible with functions like get_cluster_assignment_probabilities().
        
        Parameters
        ----------
        level : int
            Hierarchy level to get cluster specifications for
            
        Returns
        -------
        List[Tuple[int, int]]
            List of (level, cluster_id) tuples for all clusters at the specified level.
            Each tuple specifies a cluster that can be used with other functions.
            
        Raises
        ------
        ValueError
            If the specified level is not found in the hierarchical result, or if
            hierarchical clustering hasn't been performed yet
            
        Examples
        --------
        >>> distance_mat = CoassociationDistanceMatrix(clusterer, n_splits=10)
        >>> distance_mat.fit(X)
        >>> distance_mat.hierarchical_clustering()
        >>> 
        >>> # Get all clusters at level 2
        >>> level_2_clusters = distance_mat.get_level_clusters(level=2)
        >>> print(f"Level 2 clusters: {level_2_clusters}")
        >>> # Output: [(2, 0), (2, 1), (2, 2)] (assuming 3 clusters at level 2)
        >>> 
        >>> # Use with get_cluster_assignment_probabilities
        >>> probs = distance_mat.get_cluster_assignment_probabilities(level_2_clusters)
        >>> print(f"Probability matrix shape: {probs.shape}")
        >>> 
        >>> # Get clusters from multiple levels
        >>> level_1_clusters = distance_mat.get_level_clusters(level=1)
        >>> level_3_clusters = distance_mat.get_level_clusters(level=3)
        >>> mixed_clusters = level_1_clusters + level_3_clusters
        >>> mixed_probs = distance_mat.get_cluster_assignment_probabilities(mixed_clusters)
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
            
        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        
        if level not in cluster_assignments:
            available_levels = sorted(cluster_assignments.keys())
            raise ValueError(f"Level {level} not found. Available levels: {available_levels}")
        
        # Get all cluster IDs at this level
        labels = cluster_assignments[level]
        unique_clusters = np.unique(labels)
        
        # Create list of (level, cluster_id) tuples
        level_cluster_specs = [(level, cluster_id) for cluster_id in unique_clusters]
        
        return level_cluster_specs

    def get_cluster_assignment_probabilities(self, cluster_specs: Union[Dict, List[Tuple[int, int]]]) -> np.ndarray:
        """
        Calculate assignment probabilities for cells to a set of specified clusters.
        
        This function computes the probability that each cell belongs to each of the 
        specified clusters based on the coassociation matrix. The probability is 
        calculated as the mean coassociation strength between a cell and all cells
        currently assigned to each cluster.
        
        Parameters
        ----------
        cluster_specs : List[Tuple[int, int]]
            List of (level, cluster_id) tuples specifying the clusters of interest.
            Example: [(1, 0), (2, 1), (3, 0)] for cluster 0 at level 1, 
                     cluster 1 at level 2, and cluster 0 at level 3.
        
        Returns
        -------
        np.ndarray
            Matrix of shape (n_cells, len(cluster_specs) + 1) where:
            - Each row represents a cell
            - Columns 0 to len(cluster_specs)-1 represent probabilities for each specified cluster
            - Last column represents probability of being unassigned (1 - sum of other probabilities)
            Values are normalized so each row sums to 1.
        
        Raises
        ------
        ValueError
            If hierarchical clustering hasn't been performed, or if any specified
            level or cluster_id is not found
        
        Examples
        --------
        >>> # Get probabilities for cluster 0 at level 1 and cluster 1 at level 2
        >>> cluster_specs = [(1, 0), (2, 1)]
        >>> probs = coassoc.get_cluster_assignment_probabilities(cluster_specs)
        >>> print(f"Cell 0 has {probs[0, 0]:.3f} probability for cluster (1,0)")
        >>> print(f"Cell 0 has {probs[0, 1]:.3f} probability for cluster (2,1)")
        >>> print(f"Cell 0 has {probs[0, 2]:.3f} probability of being unassigned")
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
        
        if type(cluster_specs) in [list]:
            cluster_specs_dict = {c:i for i,c in enumerate(cluster_specs)}
        else:
            cluster_specs_dict = cluster_specs

        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        coassoc_matrix = self.get_coassociation_matrix()

        # Validate all cluster specifications
        cluster_indices_list = pd.DataFrame()
        total = np.zeros((coassoc_matrix.shape[0], 1), dtype=float)
        for v in np.unique(list(cluster_specs_dict.values())):
            mask = np.zeros(coassoc_matrix.shape[0], dtype=bool)
            for (level, cluster_id), v_ in cluster_specs_dict.items():
                
                if v_ != v:
                    continue

                if level not in cluster_assignments:
                    available_levels = sorted(cluster_assignments.keys())
                    raise ValueError(f"Level {level} not found. Available levels: {available_levels}")

                labels = cluster_assignments[level]

                available_clusters = np.unique(labels)
                
                if cluster_id not in available_clusters:
                    raise ValueError(f"Cluster {cluster_id} not found at level {level}. "
                                f"Available clusters: {sorted(available_clusters)}")
            
                # Get indices of cells in this cluster
                mask += labels == cluster_id

            print(v, mask)
            w = coassoc_matrix * mask.reshape(-1, 1)
            print(w.shape)
            cluster_indices_list.loc[:,v] = w.flatten()/mask.sum()
            total += w/mask.sum()

        # Normalize    
        cluster_indices_list /= total
        
        return cluster_indices_list
    
    def get_partition_assignment_probabilities(self, target_partition: np.ndarray, 
                                             similarity_metric: str = 'ari') -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        """
        Find best matching clusters for a given partition and compute assignment probabilities.
        
        This function takes a target partition and finds the best corresponding clusters 
        from the hierarchical tree using similarity metrics like ARI. It then computes 
        assignment probabilities based on those matched clusters.
        
        Parameters
        ----------
        target_partition : np.ndarray
            Array of cluster labels for the target partition, shape (n_cells,).
            Should have the same length as the original data.
        similarity_metric : str, default='ari'
            Similarity metric to use for matching. Options:
            - 'ari': Adjusted Rand Index
            - 'ami': Adjusted Mutual Information  
            - 'nmi': Normalized Mutual Information
            - 'jaccard': Jaccard similarity
        
        Returns
        -------
        Tuple[np.ndarray, List[Tuple[int, int, float]]]
            - prob_matrix: Assignment probability matrix of shape (n_cells, n_target_clusters + 1)
              where the last column is unassigned probability
            - best_matches: List of (level, cluster_id, similarity_score) for each target cluster
        
        Raises
        ------
        ValueError
            If hierarchical clustering hasn't been performed, or if target_partition 
            has wrong length, or if similarity_metric is not supported
        
        Examples
        --------
        >>> # Compare with a given partition (e.g., true labels)
        >>> probs, matches = coassoc.get_partition_assignment_probabilities(true_labels, 'ari')
        >>> print(f"Best matches: {matches}")
        >>> print(f"Cell 0 probabilities: {probs[0]}")
        """
        if self.hierarchical_results_ is None:
            raise ValueError("Hierarchical clustering has not been performed yet. Call hierarchical_clustering() first.")
        
        # Import similarity metrics (sklearn is required dependency)
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
        
        cluster_assignments = self.hierarchical_results_.get('cluster_assignments', {})
        
        # Validate target partition
        if len(target_partition) != len(self.get_cluster_labels(list(cluster_assignments.keys())[0])):
            raise ValueError(f"Target partition length ({len(target_partition)}) doesn't match data size")
        
        # Define similarity function
        if similarity_metric == 'ari':
            similarity_func = adjusted_rand_score
        elif similarity_metric == 'ami':
            similarity_func = adjusted_mutual_info_score
        elif similarity_metric == 'nmi':
            similarity_func = normalized_mutual_info_score
        elif similarity_metric == 'jaccard':
            from sklearn.metrics import jaccard_score
            def jaccard_similarity(a, b):
                # Convert to binary matrices and compute Jaccard
                # For multiclass, use macro average
                return jaccard_score(a, b, average='macro', zero_division=0)
            similarity_func = jaccard_similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}. "
                           f"Use 'ari', 'ami', 'nmi', or 'jaccard'")
        
        # Get unique clusters in target partition
        target_clusters = sorted(np.unique(target_partition))
        n_target_clusters = len(target_clusters)
        
        # Find best matching cluster for each target cluster
        best_matches = []
        
        for target_cluster_id in target_clusters:
            target_mask = (target_partition == target_cluster_id)
            best_similarity = -1
            best_level = None
            best_cluster = None
            
            # Search through all levels and clusters
            for level in cluster_assignments.keys():
                level_labels = cluster_assignments[level]
                level_clusters = sorted(np.unique(level_labels))
                
                for cluster_id in level_clusters:
                    cluster_mask = (level_labels == cluster_id)
                    
                    # Create binary labels for similarity comparison
                    target_binary = target_mask.astype(int)
                    cluster_binary = cluster_mask.astype(int)
                    
                    # Calculate similarity
                    similarity = similarity_func(target_binary, cluster_binary)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_level = level
                        best_cluster = cluster_id
            
            best_matches.append((best_level, best_cluster, best_similarity))
        
        # Create cluster specifications for probability calculation
        cluster_specs = [(level, cluster_id) for level, cluster_id, _ in best_matches]
        
        # Calculate assignment probabilities using the matched clusters
        prob_matrix = self.get_cluster_assignment_probabilities(cluster_specs)
        
        return prob_matrix, best_matches
    
    def analyze_partition_correspondence(self, target_partition: np.ndarray, 
                                       similarity_metric: str = 'ari',
                                       min_similarity: float = 0.1) -> Dict[str, Any]:
        """
        Comprehensive analysis of how a target partition corresponds to the hierarchical tree.
        
        Parameters
        ----------
        target_partition : np.ndarray
            Array of cluster labels for the target partition
        similarity_metric : str, default='ari'
            Similarity metric to use ('ari', 'ami', 'nmi', 'jaccard')
        min_similarity : float, default=0.1
            Minimum similarity threshold to report a match
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'probabilities': Assignment probability matrix
            - 'best_matches': List of best matches for each target cluster
            - 'match_quality': Overall quality metrics
            - 'level_coverage': Which hierarchical levels were used
            - 'unmatched_clusters': Target clusters with low similarity matches
        """
        prob_matrix, best_matches = self.get_partition_assignment_probabilities(
            target_partition, similarity_metric)
        
        # Analyze match quality
        similarities = [match[2] for match in best_matches]
        match_quality = {
            'mean_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_similarity': np.std(similarities)
        }
        
        # Analyze level coverage
        used_levels = list(set(match[0] for match in best_matches))
        level_coverage = {
            'used_levels': sorted(used_levels),
            'n_levels_used': len(used_levels),
            'level_distribution': {level: sum(1 for match in best_matches if match[0] == level) 
                                 for level in used_levels}
        }
        
        # Find unmatched clusters (low similarity)
        target_clusters = sorted(np.unique(target_partition))
        unmatched_clusters = []
        for i, (target_cluster, (level, cluster_id, similarity)) in enumerate(zip(target_clusters, best_matches)):
            if similarity < min_similarity:
                unmatched_clusters.append({
                    'target_cluster': target_cluster,
                    'best_match': (level, cluster_id),
                    'similarity': similarity,
                    'n_cells': np.sum(target_partition == target_cluster)
                })
        
        return {
            'probabilities': prob_matrix,
            'best_matches': best_matches,
            'match_quality': match_quality,
            'level_coverage': level_coverage,
            'unmatched_clusters': unmatched_clusters,
            'target_clusters': target_clusters
        }