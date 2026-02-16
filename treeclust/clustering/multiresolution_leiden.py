"""
Multiresolution Leiden clustering for treeclust.

This module provides a MultiresolutionLeiden class that runs Leiden clustering
across multiple resolution values to explore different scales of community detection.
"""

import numpy as np
import warnings
from typing import Any, Dict, List, Union, Optional
from scipy import sparse
import scipy.sparse as sp
import copy

from .leiden import Leiden
from tqdm import tqdm

class MultiresolutionLeiden:
    """
    Multiresolution Leiden clustering class for resolution parameter tuning.
    
    This class runs Leiden clustering across multiple resolution values to explore
    community detection at different scales. It provides comprehensive results
    analysis and consistency checking for each resolution.
    
    Features:
    - Runs Leiden clustering across multiple resolution values
    - Stores clustering results and metrics for all resolutions
    - Computes consistency scores when n_repetitions > 1
    - Provides methods to find optimal resolution based on different criteria
    - sklearn-style interface for easy integration
    - Support for both dense and sparse adjacency matrices
    """
    
    def __init__(
        self,
        resolution_values: Union[List, np.ndarray, tuple] = None,
        resolution_range: Optional[tuple] = (0, 1),
        random_state: int = 0,
        partition_type: str = 'RB',
        flavor: str = 'auto',
        n_repetitions: int = 10,
        consistency_metric: str = 'ari',
        store_clusterers: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the MultiresolutionLeiden clustering class.
        
        Parameters:
        -----------
        resolution_values : list, array-like, or None, default=None
            Specific resolution values to test for Leiden clustering.
            Higher values generally lead to more clusters.
            Mutually exclusive with resolution_range.
            
        resolution_range : tuple or None, default=None
            Resolution range (min_resolution, max_resolution) for automatic
            resolution profile generation using leidenalg.Optimiser.
            When provided, automatically finds optimal resolution values
            where partition changes occur. Mutually exclusive with resolution_values.
            
        random_state : int, default=0
            Random seed for reproducible clustering.
            
        partition_type : str, default='RB'
            Partition method to use for Leiden clustering.
            
        flavor : str, default='auto'
            Leiden algorithm implementation to use.
            
        n_repetitions : int, default=10
            Number of repetitions to run for each resolution for consistency checking.
            
        consistency_metric : str, default='ari'
            Metric to use for measuring consistency between repetitions.
            
        store_clusterers : bool, default=True
            Whether to store the fitted Leiden objects for each resolution.
            If False, only stores the cluster labels to save memory.
            
        verbose : bool, default=True
            Whether to show progress bar during multiresolution clustering.
            
        Examples:
        --------
        >>> from treeclust.clustering import MultiresolutionLeiden
        >>> from treeclust.neighbors import KNeighbors
        >>> 
        >>> # Create adjacency matrix
        >>> knn = KNeighbors(n_neighbors=15, mode='connectivity')
        >>> adjacency = knn.fit_transform(X)
        >>> 
        >>> # Test Leiden across different resolutions
        >>> mr_leiden = MultiresolutionLeiden(
        ...     resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0],
        ...     n_repetitions=5
        ... )
        >>> results = mr_leiden.fit_predict(adjacency)
        >>> 
        >>> # Find best resolution based on number of clusters
        >>> best_resolution = mr_leiden.get_best_resolution('n_clusters')
        >>> best_labels = mr_leiden.get_labels(best_resolution)
        >>> 
        >>> # Or use resolution range for automatic profile generation
        >>> mr_leiden_auto = MultiresolutionLeiden(
        ...     resolution_range=(0, 2.0),
        ...     n_repetitions=5
        ... )
        >>> results_auto = mr_leiden_auto.fit_predict(adjacency)
        """
        # Validate parameters
        if resolution_values is not None and resolution_range is not None:
            raise ValueError("Cannot specify both resolution_values and resolution_range")
        
        if resolution_values is None and resolution_range is None:
            raise ValueError("Must specify either resolution_values or resolution_range")
        
        if resolution_range is not None:
            if not isinstance(resolution_range, tuple) or len(resolution_range) != 2:
                raise ValueError("resolution_range must be a tuple of (min_resolution, max_resolution)")
            if resolution_range[0] >= resolution_range[1]:
                raise ValueError("resolution_range[0] must be less than resolution_range[1]")
        
        self.resolution_values = np.array(resolution_values) if resolution_values is not None else None
        self.resolution_range = resolution_range
        self.random_state = int(random_state)
        self.partition_type = partition_type
        self.flavor = flavor
        self.n_repetitions = int(n_repetitions)
        self.consistency_metric = consistency_metric
        self.store_clusterers = store_clusterers
        self.verbose = verbose
        
        # Validate inputs
        if self.resolution_values is not None and len(self.resolution_values) == 0:
            raise ValueError("resolution_values must contain at least one value")
        
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be >= 1")
        
        # Storage for results
        self.clusterers_ = {} if store_clusterers else None
        self.labels_ = {}  # Now indexed by resolution order (0, 1, 2, ...)
        self.resolution_values_ = {}  # Maps index to actual resolution value (0->0.1, 1->0.5, etc.)
        self.metrics_ = {}
        self.consistency_summaries_ = {}
        self.cell_confidence_ = {}  # Per-cell confidence scores for each resolution
        self.is_fitted_ = False
        self.data_shape_ = None
        
    def _generate_resolution_profile(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        """
        Generate resolution profile using leidenalg.Optimiser.
        
        Parameters:
        -----------
        adjacency_matrix : array-like
            Adjacency matrix of the graph.
            
        Returns:
        --------
        resolution_values : np.ndarray
            Array of resolution values where partition changes occur.
        """
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError:
            raise ImportError("igraph and leidenalg are required for resolution profile generation. "
                            "Please install with: pip install python-igraph leidenalg")
        
        # Convert adjacency matrix to igraph
        if sp.issparse(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.tocoo()
            edges = list(zip(adjacency_matrix.row, adjacency_matrix.col))
            weights = adjacency_matrix.data
        else:
            # Convert dense matrix to edge list
            rows, cols = np.nonzero(adjacency_matrix)
            edges = list(zip(rows, cols))
            weights = adjacency_matrix[rows, cols]
        
        # Create igraph Graph
        G = ig.Graph(edges=edges, directed=False)
        G.es['weight'] = weights
        
        # Get partition type class
        if self.partition_type == 'RB':
            partition_class = la.RBConfigurationVertexPartition
        elif self.partition_type == 'CPM':
            partition_class = la.CPMVertexPartition
        elif self.partition_type == 'Modularity':
            partition_class = la.ModularityVertexPartition
        else:
            raise ValueError(f"Unsupported partition type: {self.partition_type}")
        
        # Generate resolution profile
        optimiser = la.Optimiser()
        
        # Set random seed for reproducible resolution profile generation
        optimiser.set_rng_seed(self.random_state)
        
        if self.verbose:
            print(f"Generating resolution profile from {self.resolution_range[0]} to {self.resolution_range[1]}...")
        
        profile = optimiser.resolution_profile(G, partition_class, 
                                             resolution_range=self.resolution_range,
                                             number_iterations=10
                                             )
        
        # Extract transition boundary values from profile
        transition_boundaries = [partition.resolution_parameter for partition in profile]
        transition_boundaries = sorted(set(transition_boundaries))  # Remove duplicates and sort
        
        # Generate stable resolution values between transition boundaries
        stable_resolutions = []
        
        # Add the minimum resolution (start of range) as a stable point
        min_res, max_res = self.resolution_range
        stable_resolutions.append(min_res)
        
        # Add midpoints between consecutive transitions for stability
        for i in range(len(transition_boundaries) - 1):
            midpoint = (transition_boundaries[i] + transition_boundaries[i + 1]) / 2
            stable_resolutions.append(midpoint)
        
        # Add the maximum resolution (end of range) as a stable point
        if len(transition_boundaries) > 0:
            # Add midpoint between last transition and max resolution
            final_midpoint = (transition_boundaries[-1] + max_res) / 2
            stable_resolutions.append(final_midpoint)
        
        stable_resolutions.append(max_res)
        
        # Remove duplicates and sort
        resolution_values = sorted(set(stable_resolutions))
        
        if self.verbose:
            print(f"Found {len(transition_boundaries)} transition boundaries")
            print(f"Generated {len(resolution_values)} stable resolution values")
        
        return np.array(resolution_values)
        
    def fit(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> 'MultiresolutionLeiden':
        """
        Fit Leiden clustering for all resolution values.
        
        Parameters:
        -----------
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Adjacency matrix of shape (n_samples, n_samples).
            
        Returns:
        --------
        self : MultiresolutionLeiden
            Returns self for method chaining.
        """
        self.data_shape_ = adjacency_matrix.shape
        
        # Determine resolution values to use
        if self.resolution_range is not None:
            # Generate resolution profile automatically
            resolution_values = self._generate_resolution_profile(adjacency_matrix)
        else:
            # Use provided resolution values
            resolution_values = self.resolution_values
        
        # Clear previous results
        if self.store_clusterers:
            self.clusterers_ = {}
        self.labels_ = {}
        self.resolution_values_ = {}
        self.metrics_ = {}
        self.consistency_summaries_ = {}
        self.cell_confidence_ = {}
        
        # Sort resolution values and create index mapping
        sorted_resolutions = sorted(resolution_values)
        for index, resolution_value in enumerate(sorted_resolutions):
            self.resolution_values_[index] = resolution_value
        
        # Fit Leiden clustering for each resolution value with optional progress bar
        resolution_iter = sorted_resolutions
        if self.verbose:
            resolution_iter = tqdm(resolution_iter, 
                                 desc="Multiresolution Leiden", 
                                 unit="resolution")
        
        for resolution in resolution_iter:
            # Get the index for this resolution
            index = next(i for i, r in self.resolution_values_.items() if r == resolution)
            
            try:
                # Create Leiden instance with current resolution
                leiden = Leiden(
                    resolution=resolution,
                    random_state=self.random_state,
                    partition_type=self.partition_type,
                    flavor=self.flavor,
                    n_repetitions=self.n_repetitions,
                    consistency_metric=self.consistency_metric
                )
                
                # Fit the clusterer
                leiden.fit(adjacency_matrix)
                
                # Store results using index instead of resolution value
                if self.store_clusterers:
                    self.clusterers_[index] = leiden
                
                # Get cluster labels and store by index
                labels = leiden.labels_
                self.labels_[index] = labels
                
                # Compute basic metrics and store by index
                self.metrics_[index] = self._compute_metrics(labels, adjacency_matrix)
                
                # Store consistency summary if multiple repetitions
                if self.n_repetitions > 1:
                    self.consistency_summaries_[index] = leiden.get_consistency_summary()
                    # Store cell confidence scores
                    self.cell_confidence_[index] = leiden.cell_confidence_
                else:
                    self.cell_confidence_[index] = None
                
            except Exception as e:
                warnings.warn(f"Failed to fit Leiden with resolution={resolution}: {e}")
                self.labels_[index] = None
                self.metrics_[index] = None
                if self.n_repetitions > 1:
                    self.consistency_summaries_[index] = None
        
        self.is_fitted_ = True
        return self
        
    def fit_predict(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> Dict[float, np.ndarray]:
        """
        Fit clustering models and return cluster labels for all resolution values.
        
        Parameters:
        -----------
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Adjacency matrix of shape (n_samples, n_samples).
            
        Returns:
        --------
        labels_dict : dict
            Dictionary mapping resolution values to cluster labels arrays.
        """
        self.fit(adjacency_matrix)
        return self.labels_.copy()

    def get_observed_clusters(self, resolution_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get cluster assignments for a specific resolution ID. If resolution_id is None,
        returns cluster assignments for all resolutions as a dictionary.

        Parameters:
        -----------
        resolution_id : int or None, default=None
            The resolution index to get clusters for. If None, returns all.

        Returns:
        --------
        clusters : np.ndarray or dict or None
            Cluster assignments array for the specified resolution index,
            or a dictionary of all cluster assignments if resolution_id is None.
        """

        if not self.is_fitted_:
            raise ValueError("MultiresolutionLeiden has not been fitted yet. Call fit() first.")
        
        if resolution_id is not None:
            return self.labels_.get(resolution_id, None).copy()
        else:
            return self.labels_.copy()

    def get_observed_confidence(self, resolution_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get per-observation confidence scores for a specific resolution ID. If resolution_id is None,
        returns confidence scores for all resolutions as a dictionary.

        Parameters:
        -----------
        resolution_id : int or None, default=None
            The resolution index to get confidence scores for. If None, returns all.

        Returns:
        --------
        confidence_scores : np.ndarray or dict or None
            Confidence scores array for the specified resolution index,
            or a dictionary of all confidence scores if resolution_id is None.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLeiden has not been fitted yet. Call fit() first.")
        
        if resolution_id is not None:
            return self.cell_confidence_.get(resolution_id, None).copy()
        else:
            return self.cell_confidence_.copy()

    def get_cluster_metric(self, resolution_id: Optional[int] = None, metric: str = "stability") -> Optional[float]:
        """
        Get the specified metric for a specific cluster at a given resolution index.
        If resolution_index is None, returns metrics for all clusters at all resolutions.

        Parameters:
        -----------
        resolution_id : Optional[int], default=None
            Resolution index (0, 1, 2, ...)

        metric : str, default="stability"
            The metric to retrieve (e.g., 'stability').

        Returns:
        --------
        dict or None
            A dictionary of stability scores for all clusters at the specified resolution,
            or a nested dictionary of all stability scores if resolution_index is None.
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if resolution_id is not None:
            if resolution_id not in self.consistency_summaries_:
                raise ValueError(f"Resolution index {resolution_id} not found. Available: {list(self.consistency_summaries_.keys())}")
            summary = self.consistency_summaries_[resolution_id]
            if summary is None:
                return None
            return summary["per_cluster_consistency"].copy()
        else:
            all_stabilities = {}
            for index, summary in self.consistency_summaries_.items():
                for i, v in summary["per_cluster_consistency"].items():
                    all_stabilities[(index, i)] = v[metric]
            return all_stabilities

    def get_resolution_range_summary(self) -> Dict[str, Any]:
        """
        Get a summary of results across all resolution values.
        
        Returns:
        --------
        summary : dict
            Summary statistics including resolution range, cluster counts, success rates, etc.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLeiden has not been fitted yet. Call fit() first.")
        
        # Get actual resolution values used (from resolution_values_ mapping)
        actual_resolutions = list(self.resolution_values_.values())
        
        successful = sum(1 for labels in self.labels_.values() if labels is not None)
        failed = len(actual_resolutions) - successful
        
        n_clusters_list = []
        consistency_list = []
        
        for index in range(len(self.resolution_values_)):
            # Collect cluster counts
            metrics = self.metrics_.get(index, {})
            if metrics and 'n_clusters' in metrics:
                n_clusters_list.append(metrics['n_clusters'])
            
            # Collect consistency scores
            if self.n_repetitions > 1:
                summary = self.consistency_summaries_.get(index, {})
                if summary and summary.get('mean_consistency') is not None:
                    consistency_list.append(summary['mean_consistency'])
        
        result = {
            'resolution_values': sorted(actual_resolutions),
            'successful_fits': successful,
            'failed_fits': failed,
            'total_resolutions': len(actual_resolutions),
            'n_repetitions': self.n_repetitions,
            'n_clusters': n_clusters_list,
            'consistencies': consistency_list
        }
        
        if n_clusters_list:
            result['n_clusters_range'] = [min(n_clusters_list), max(n_clusters_list)]
        else:
            result['n_clusters_range'] = None
            result['best_resolution_clusters'] = None
        
        if consistency_list:
            result['consistency_range'] = [min(consistency_list), max(consistency_list)]
            result['mean_consistency'] = np.mean(consistency_list)
        else:
            result['consistency_range'] = None
            result['mean_consistency'] = None
            
        return result
        
    def _compute_metrics(self, labels: np.ndarray, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Parameters:
        -----------
        labels : np.ndarray
            Cluster labels.
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Input adjacency matrix.
            
        Returns:
        --------
        metrics : dict
            Dictionary of computed metrics.
        """
        if labels is None:
            return {}
        
        metrics = {}
        
        # Basic metrics
        unique_labels = np.unique(labels)
        metrics['n_clusters'] = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        metrics['n_noise_points'] = np.sum(labels == -1) if np.any(labels == -1) else 0
        
        # Modularity (since we have adjacency matrix)
        try:
            metrics['modularity'] = self._compute_modularity(adjacency_matrix, labels)
        except Exception:
            pass
        
        return metrics
        
    def _compute_modularity(self, adjacency: Union[np.ndarray, sp.spmatrix], 
                           labels: np.ndarray) -> float:
        """
        Compute modularity for given adjacency matrix and cluster labels.
        
        Parameters:
        -----------
        adjacency : np.ndarray or scipy.sparse matrix
            Adjacency matrix.
        labels : np.ndarray
            Cluster labels.
            
        Returns:
        --------
        modularity : float
            Modularity score.
        """
        try:
            import igraph as ig
            
            # Convert to igraph format
            if sparse.issparse(adjacency):
                sources, targets = adjacency.nonzero()
                weights = adjacency.data
            else:
                sources, targets = np.nonzero(adjacency)
                weights = adjacency[sources, targets]
            
            # Create igraph graph
            n_vertices = adjacency.shape[0]
            g = ig.Graph()
            g.add_vertices(n_vertices)
            
            # Add edges
            edges = list(zip(sources.tolist(), targets.tolist()))
            g.add_edges(edges)
            g.es['weight'] = weights.tolist()
            
            # Compute modularity
            return g.modularity(labels.tolist(), weights='weight')
            
        except ImportError:
            # Fallback: simple modularity computation without igraph
            return self._compute_modularity_simple(adjacency, labels)
            
    def _compute_modularity_simple(self, adjacency: Union[np.ndarray, sp.spmatrix], 
                                  labels: np.ndarray) -> float:
        """
        Simple modularity computation without igraph dependency.
        """
        if sparse.issparse(adjacency):
            A = adjacency
        else:
            A = sparse.csr_matrix(adjacency)
        
        n = A.shape[0]
        m = A.sum() / 2.0  # Total number of edges
        
        if m == 0:
            return 0.0
        
        # Compute degree
        degrees = np.array(A.sum(axis=1)).flatten()
        
        # Compute modularity
        modularity = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            mask = labels == label
            indices = np.where(mask)[0]
            
            if len(indices) < 2:
                continue
            
            # Sum of weights within community
            A_sub = A[np.ix_(indices, indices)]
            l_c = A_sub.sum() / 2.0
            
            # Sum of degrees in community
            d_c = degrees[indices].sum()
            
            # Modularity contribution
            modularity += (l_c / m) - (d_c / (2.0 * m)) ** 2
        
        return modularity
    
    def predict(self, 
                annotations: Dict[tuple, Union[str, int, float]], 
                default_value: Union[str, int, float] = None) -> np.ndarray:
        """
        Predict labels for all cells based on cluster annotations.
        
        Takes a dictionary specifying (resolution_index, cluster) -> label mappings
        and predicts labels for all cells. Uses confidence scores as weights when
        cells belong to multiple annotated clusters.
        
        Parameters
        ----------
        annotations : dict
            Dictionary mapping (resolution_index, cluster_id) tuples to label values.
            Example: {(0, 1): 'T_cell', (0, 2): 'B_cell', (1, 3): 'T_cell'}
            
        default_value : str, int, float, or None, default=None
            Value to assign to cells that don't have annotations or have ties.
            If None and annotations contain strings, uses 'unassigned'.
            If None and annotations contain integers, uses np.nan.
            
        Returns
        -------
        np.ndarray
            Array of predictions for all cells.
            
        Examples
        --------
        >>> # Annotate some clusters with cell types
        >>> annotations = {
        ...     (0, 1): 'T_cell',
        ...     (0, 2): 'B_cell', 
        ...     (1, 3): 'T_cell',
        ...     (2, 0): 'NK_cell'
        ... }
        >>> predictions = mleiden.predict(annotations)
        """
        if not self.is_fitted_:
            raise ValueError("Must fit the model before making predictions")
            
        # Get number of cells from any available labels
        n_cells = None
        for labels in self.labels_.values():
            if labels is not None:
                n_cells = len(labels)
                break
                
        if n_cells is None:
            raise ValueError("No fitted labels found")
            
        # Determine default value if not provided
        if default_value is None:
            if annotations:
                sample_value = next(iter(annotations.values()))
                if isinstance(sample_value, str):
                    default_value = 'unassigned'
                else:
                    default_value = np.nan
            else:
                default_value = np.nan
        
        # Initialize arrays to track votes and weights
        votes = {}  # label -> array of weights for each cell
        
        # Process each annotation
        for (res_idx, cluster_id), label in annotations.items():
            if res_idx not in self.labels_:
                warnings.warn(f"Resolution index {res_idx} not found in fitted results")
                continue
                
            labels = self.labels_[res_idx]
            if labels is None:
                warnings.warn(f"No labels found for resolution index {res_idx}")
                continue
                
            # Find cells in this cluster
            labels_array = np.array(labels)
            cluster_mask = (labels_array == cluster_id)
            if not np.any(cluster_mask):
                warnings.warn(f"Cluster {cluster_id} not found in resolution {res_idx}")
                continue
                
            # Get confidence scores for this resolution (use 1.0 if not available)
            if res_idx in self.cell_confidence_:
                confidence_scores = self.cell_confidence_[res_idx]
                weights = np.where(cluster_mask, confidence_scores, 0.0)
            else:
                weights = np.where(cluster_mask, 1.0, 0.0)
            
            # Add votes for this label
            if label not in votes:
                votes[label] = np.zeros(n_cells)
            votes[label] += weights
        
        # Determine final predictions
        predictions = np.full(n_cells, default_value, dtype=object)
        
        if votes:
            # Calculate unassigned probabilities: max(1 - sum(other_probabilities), 0)
            unassigned_weights = np.ones(n_cells)  # Start with weight 1 for everyone
            
            # Subtract evidence for each annotated class
            for label, weight_array in votes.items():
                unassigned_weights -= weight_array
            
            # Clip to [0, inf) - unassigned weight is max(1 - total_evidence, 0)
            unassigned_weights = np.maximum(unassigned_weights, 0.0)
            
            # Add unassigned as a voting option
            votes[default_value] = unassigned_weights
            
            # For each cell, find the label with maximum weight
            for i in range(n_cells):
                max_weight = 0.0
                best_label = default_value
                tied = False
                
                for label, weight_array in votes.items():
                    weight = weight_array[i]
                    if weight > max_weight:
                        max_weight = weight
                        best_label = label
                        tied = False
                    elif weight > 0 and weight == max_weight:
                        tied = True
                
                if max_weight > 0 and not tied:
                    predictions[i] = best_label
                # If tied or no votes, keep default_value
        
        return predictions
    
    def predict_proba(self, 
                      annotations: Dict[tuple, Union[str, int, float]], 
                      normalize: bool = True) -> tuple:
        """
        Predict class probabilities for all cells based on cluster annotations.
        
        Creates a soft assignment matrix showing the probability of each cell
        belonging to each annotated class, weighted by confidence scores.
        
        Parameters
        ----------
        annotations : dict
            Dictionary mapping (resolution_index, cluster_id) tuples to label values.
            Example: {(0, 1): 'T_cell', (0, 2): 'B_cell', (1, 3): 'T_cell'}
            
        normalize : bool, default=True
            Whether to normalize probabilities so each cell sums to 1.
            If False, returns raw weighted scores.
            
        Returns
        -------
        probabilities : np.ndarray
            Array of shape (n_cells, n_classes) with probability/weight for each
            cell-class combination.
            
        class_names : list
            List of class names corresponding to columns in probabilities array.
            
        Examples
        --------
        >>> annotations = {
        ...     (0, 1): 'T_cell',
        ...     (0, 2): 'B_cell', 
        ...     (1, 3): 'T_cell'
        ... }
        >>> probs, classes = mleiden.predict_proba(annotations)
        >>> print(f"Classes: {classes}")
        >>> print(f"Cell 0 probabilities: {probs[0]}")
        """
        if not self.is_fitted_:
            raise ValueError("Must fit the model before making predictions")
            
        # Get number of cells from any available labels
        n_cells = None
        for labels in self.labels_.values():
            if labels is not None:
                n_cells = len(labels)
                break
                
        if n_cells is None:
            raise ValueError("No fitted labels found")
            
        # Get unique class names and add unassigned class
        class_names = sorted(set(annotations.values()))
        
        # Determine unassigned class name
        sample_value = next(iter(annotations.values()))
        if isinstance(sample_value, str):
            unassigned_name = 'unassigned'
        else:
            unassigned_name = np.nan
            
        # Add unassigned to class names if not already present
        if unassigned_name not in class_names:
            class_names.append(unassigned_name)
            
        n_classes = len(class_names)
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        # Initialize probability matrix
        probabilities = np.zeros((n_cells, n_classes))
        
        # Process each annotation
        for (res_idx, cluster_id), label in annotations.items():
            if res_idx not in self.labels_:
                warnings.warn(f"Resolution index {res_idx} not found in fitted results")
                continue
                
            labels = self.labels_[res_idx]
            if labels is None:
                warnings.warn(f"No labels found for resolution index {res_idx}")
                continue
                
            # Find cells in this cluster
            labels_array = np.array(labels)
            cluster_mask = (labels_array == cluster_id)
            if not np.any(cluster_mask):
                warnings.warn(f"Cluster {cluster_id} not found in resolution {res_idx}")
                continue
                
            # Get confidence scores for this resolution (use 1.0 if not available)
            if res_idx in self.cell_confidence_:
                confidence_scores = self.cell_confidence_[res_idx]
                weights = np.where(cluster_mask, confidence_scores, 0.0)
            else:
                weights = np.where(cluster_mask, 1.0, 0.0)
            
            # Add weights to appropriate class column
            class_idx = class_to_idx[label]
            probabilities[:, class_idx] += weights
        
        # Calculate unassigned probabilities: max(1 - sum(other_probabilities), 0)
        if unassigned_name in class_to_idx:
            unassigned_idx = class_to_idx[unassigned_name]
            
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
        
        return probabilities, class_names
        
    def __repr__(self) -> str:
        """String representation of the MultiresolutionLeiden object."""
        if self.resolution_values is not None:
            n_res = len(self.resolution_values)
            res_range = f"[{self.resolution_values.min():.3f}, {self.resolution_values.max():.3f}]"
        elif self.resolution_range is not None:
            n_res = f"range{self.resolution_range}"
            res_range = f"{self.resolution_range}"
        else:
            n_res = "unknown"
            res_range = "unknown"
            
        return (f"MultiresolutionLeiden("
                f"n_resolutions={n_res}, "
                f"resolution_range={res_range}, "
                f"partition_type='{self.partition_type}', "
                f"n_repetitions={self.n_repetitions}, "
                f"consistency_metric='{self.consistency_metric}', "
                f"store_clusterers={self.store_clusterers}, "
                f"fitted={self.is_fitted_})")