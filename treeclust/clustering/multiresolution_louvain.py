"""
Multiresolution Louvain clustering for treeclust.

This module provides a MultiresolutionLouvain class that runs Louvain clustering
across multiple resolution values to explore different scales of community detection.
"""

import numpy as np
import warnings
from typing import Any, Dict, List, Union, Optional
from scipy import sparse
import scipy.sparse as sp
import copy

from .louvain import Louvain
from tqdm import tqdm


class MultiresolutionLouvain:
    """
    Multiresolution Louvain clustering class for resolution parameter tuning.
    
    This class runs Louvain clustering across multiple resolution values to explore
    community detection at different scales. It provides comprehensive results
    analysis and consistency checking for each resolution.
    
    Features:
    - Runs Louvain clustering across multiple resolution values
    - Stores clustering results and metrics for all resolutions
    - Computes consistency scores when n_repetitions > 1
    - Provides methods to find optimal resolution based on different criteria
    - sklearn-style interface for easy integration
    - Support for both dense and sparse adjacency matrices
    """
    
    def __init__(
        self,
        resolution_values: Union[List, np.ndarray],
        random_state: int = 0,
        flavor: str = 'auto',
        n_repetitions: int = 10,
        consistency_metric: str = 'ari',
        store_clusterers: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the MultiresolutionLouvain clustering class.
        
        Parameters:
        -----------
        resolution_values : list or array-like
            Resolution values to test for Louvain clustering.
            Higher values generally lead to more clusters.
            
        random_state : int, default=0
            Random seed for reproducible clustering.
            
        flavor : str, default='auto'
            Louvain algorithm implementation to use.
            
        n_repetitions : int, default=10
            Number of repetitions to run for each resolution for consistency checking.
            
        consistency_metric : str, default='ari'
            Metric to use for measuring consistency between repetitions.
            
        store_clusterers : bool, default=True
            Whether to store the fitted Louvain objects for each resolution.
            If False, only stores the cluster labels to save memory.
            
        verbose : bool, default=True
            Whether to show progress bar during multiresolution clustering.
            
        Examples:
        --------
        >>> from treeclust.clustering import MultiresolutionLouvain
        >>> from treeclust.neighbors import KNeighbors
        >>> 
        >>> # Create adjacency matrix
        >>> knn = KNeighbors(n_neighbors=15, mode='connectivity')
        >>> adjacency = knn.fit_transform(X)
        >>> 
        >>> # Test Louvain across different resolutions
        >>> mr_louvain = MultiresolutionLouvain(
        ...     resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0],
        ...     n_repetitions=5
        ... )
        >>> results = mr_louvain.fit_predict(adjacency)
        >>> 
        >>> # Find best resolution based on number of clusters
        >>> best_resolution = mr_louvain.get_best_resolution('n_clusters')
        >>> best_labels = mr_louvain.get_labels(best_resolution)
        """
        self.resolution_values = np.array(resolution_values)
        self.random_state = int(random_state)
        self.flavor = flavor
        self.n_repetitions = int(n_repetitions)
        self.consistency_metric = consistency_metric
        self.store_clusterers = store_clusterers
        self.verbose = verbose
        
        # Validate inputs
        if len(self.resolution_values) == 0:
            raise ValueError("resolution_values must contain at least one value")
        
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be >= 1")
        
        # Storage for results
        self.clusterers_ = {} if store_clusterers else None
        self.labels_ = {}
        self.metrics_ = {}
        self.consistency_summaries_ = {}
        self.cell_confidence_ = {}  # Per-cell confidence scores for each resolution
        self.is_fitted_ = False
        self.data_shape_ = None
        
    def fit(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> 'MultiresolutionLouvain':
        """
        Fit Louvain clustering for all resolution values.
        
        Parameters:
        -----------
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Adjacency matrix of shape (n_samples, n_samples).
            
        Returns:
        --------
        self : MultiresolutionLouvain
            Returns self for method chaining.
        """
        self.data_shape_ = adjacency_matrix.shape
        
        # Calculate scaling factor: n_obs / n_edges
        n_obs = adjacency_matrix.shape[0]
        if sp.issparse(adjacency_matrix):
            n_edges = adjacency_matrix.nnz
        else:
            n_edges = np.count_nonzero(adjacency_matrix)
        
        # Avoid division by zero
        scaling_factor = n_obs / max(n_edges, 1)
        
        # Scale resolution values
        scaled_resolution_values = [res * scaling_factor for res in self.resolution_values]
        
        # Create mapping for storing results with original resolution keys
        resolution_mapping = dict(zip(scaled_resolution_values, self.resolution_values))
        
        # Clear previous results
        if self.store_clusterers:
            self.clusterers_ = {}
        self.labels_ = {}
        self.metrics_ = {}
        self.consistency_summaries_ = {}
        self.cell_confidence_ = {}
        
        # Fit Louvain clustering for each resolution value
        resolution_iter = scaled_resolution_values
        if self.verbose:
            resolution_iter = tqdm(scaled_resolution_values, desc='Multiresolution Louvain', unit='resolution')
            
        for scaled_resolution in resolution_iter:
            # Get the original resolution for storing results
            original_resolution = resolution_mapping[scaled_resolution]
            
            try:
                # Create Louvain instance with scaled resolution
                louvain = Louvain(
                    resolution=scaled_resolution,
                    random_state=self.random_state,
                    flavor=self.flavor,
                    n_repetitions=self.n_repetitions,
                    consistency_metric=self.consistency_metric
                )
                
                # Fit the clusterer
                louvain.fit(adjacency_matrix)
                
                # Store results using original resolution as key
                if self.store_clusterers:
                    self.clusterers_[original_resolution] = louvain
                
                # Get cluster labels
                labels = louvain.labels_
                self.labels_[original_resolution] = labels
                
                # Compute basic metrics
                self.metrics_[original_resolution] = self._compute_metrics(labels, adjacency_matrix)
                
                # Store consistency summary if multiple repetitions
                if self.n_repetitions > 1:
                    self.consistency_summaries_[original_resolution] = louvain.get_consistency_summary()
                    # Store cell confidence scores
                    self.cell_confidence_[original_resolution] = louvain.cell_confidence_
                else:
                    self.cell_confidence_[original_resolution] = None
                
            except Exception as e:
                warnings.warn(f"Failed to fit Louvain with resolution={original_resolution} (scaled={scaled_resolution}): {e}")
                self.labels_[original_resolution] = None
                self.metrics_[original_resolution] = None
                if self.n_repetitions > 1:
                    self.consistency_summaries_[original_resolution] = None
                    self.cell_confidence_[original_resolution] = None
                else:
                    self.cell_confidence_[original_resolution] = None
        
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
        
    def get_labels(self, resolution: float) -> Optional[np.ndarray]:
        """
        Get cluster labels for a specific resolution value.
        
        Parameters:
        -----------
        resolution : float
            The resolution value to get labels for.
            
        Returns:
        --------
        labels : np.ndarray or None
            Cluster labels array, or None if resolution not found or failed.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLouvain has not been fitted yet. Call fit() first.")
        
        return self.labels_.get(resolution, None)
        
    def get_clusterer(self, resolution: float) -> Optional[Louvain]:
        """
        Get the fitted Louvain clusterer object for a specific resolution value.
        
        Parameters:
        -----------
        resolution : float
            The resolution value to get clusterer for.
            
        Returns:
        --------
        clusterer : Louvain or None
            Fitted Louvain object, or None if not stored or resolution not found.
        """
        if not self.store_clusterers:
            raise ValueError("Clusterers were not stored. Set store_clusterers=True during initialization.")
        
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLouvain has not been fitted yet. Call fit() first.")
        
        return self.clusterers_.get(resolution, None)
        
    def get_metrics(self, resolution: Optional[float] = None) -> Union[Dict[str, float], Dict[float, Dict[str, float]]]:
        """
        Get clustering quality metrics for specific or all resolution values.
        
        Parameters:
        -----------
        resolution : float, optional
            Specific resolution value to get metrics for. If None, returns all metrics.
            
        Returns:
        --------
        metrics : dict
            Clustering metrics. If resolution is specified, returns metrics dict
            for that value. Otherwise returns dict mapping resolution values to metrics.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLouvain has not been fitted yet. Call fit() first.")
        
        if resolution is not None:
            return self.metrics_.get(resolution, {})
        else:
            return self.metrics_.copy()
            
    def get_consistency_summary(self, resolution: Optional[float] = None) -> Union[Dict[str, Any], Dict[float, Dict[str, Any]]]:
        """
        Get consistency summary for specific or all resolution values.
        
        Parameters:
        -----------
        resolution : float, optional
            Specific resolution value to get consistency for. If None, returns all.
            
        Returns:
        --------
        consistency : dict
            Consistency summary. If resolution is specified, returns summary dict
            for that value. Otherwise returns dict mapping resolution values to summaries.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLouvain has not been fitted yet. Call fit() first.")
        
        if self.n_repetitions == 1:
            raise ValueError("Consistency summaries are only available when n_repetitions > 1")
        
        if resolution is not None:
            return self.consistency_summaries_.get(resolution, {})
        else:
            return self.consistency_summaries_.copy()
            
    def get_best_resolution(self, metric: str = 'n_clusters') -> float:
        """
        Get the resolution value that optimizes a given metric.
        
        Parameters:
        -----------
        metric : str, default='n_clusters'
            Metric to optimize. Options:
            - 'n_clusters': Resolution value giving most clusters
            - 'modularity': Resolution value with highest modularity (if available)
            - 'silhouette': Resolution value with highest silhouette score (if available)
            - 'consistency': Resolution value with highest consistency (if n_repetitions > 1)
            
        Returns:
        --------
        best_resolution : float
            Resolution value that optimizes the specified metric.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLouvain has not been fitted yet. Call fit() first.")
        
        if metric == 'consistency' and self.n_repetitions == 1:
            raise ValueError("Consistency metric requires n_repetitions > 1")
        
        valid_metrics = {}
        
        if metric == 'consistency':
            for resolution, summary in self.consistency_summaries_.items():
                if summary is not None and summary.get('mean_consistency') is not None:
                    valid_metrics[resolution] = summary['mean_consistency']
        else:
            for resolution, metrics in self.metrics_.items():
                if metrics is not None and metric in metrics:
                    valid_metrics[resolution] = metrics[metric]
        
        if not valid_metrics:
            raise ValueError(f"Metric '{metric}' not found in any resolution results")
        
        # Find resolution value that maximizes the metric
        best_resolution = max(valid_metrics, key=valid_metrics.get)
        return best_resolution
        
    def get_resolution_range_summary(self) -> Dict[str, Any]:
        """
        Get a summary of results across all resolution values.
        
        Returns:
        --------
        summary : dict
            Summary statistics including resolution range, cluster counts, success rates, etc.
        """
        if not self.is_fitted_:
            raise ValueError("MultiresolutionLouvain has not been fitted yet. Call fit() first.")
        
        successful = sum(1 for labels in self.labels_.values() if labels is not None)
        failed = len(self.resolution_values) - successful
        
        n_clusters_list = []
        consistency_list = []
        
        for resolution in self.resolution_values:
            # Collect cluster counts
            metrics = self.metrics_.get(resolution, {})
            if metrics and 'n_clusters' in metrics:
                n_clusters_list.append(metrics['n_clusters'])
            
            # Collect consistency scores
            if self.n_repetitions > 1:
                summary = self.consistency_summaries_.get(resolution, {})
                if summary and summary.get('mean_consistency') is not None:
                    consistency_list.append(summary['mean_consistency'])
        
        result = {
            'resolution_values': self.resolution_values.tolist(),
            'successful_fits': successful,
            'failed_fits': failed,
            'total_resolutions': len(self.resolution_values),
            'n_repetitions': self.n_repetitions
        }
        
        if n_clusters_list:
            result['n_clusters_range'] = [min(n_clusters_list), max(n_clusters_list)]
            result['best_resolution_clusters'] = self.get_best_resolution('n_clusters')
        else:
            result['n_clusters_range'] = None
            result['best_resolution_clusters'] = None
        
        if consistency_list:
            result['consistency_range'] = [min(consistency_list), max(consistency_list)]
            result['mean_consistency'] = np.mean(consistency_list)
            result['best_resolution_consistency'] = self.get_best_resolution('consistency')
        else:
            result['consistency_range'] = None
            result['mean_consistency'] = None
            result['best_resolution_consistency'] = None
            
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
        
    def __repr__(self) -> str:
        """String representation of the MultiresolutionLouvain object."""
        return (f"MultiresolutionLouvain("
                f"n_resolutions={len(self.resolution_values)}, "
                f"resolution_range=[{self.resolution_values.min():.3f}, {self.resolution_values.max():.3f}], "
                f"flavor='{self.flavor}', "
                f"n_repetitions={self.n_repetitions}, "
                f"consistency_metric='{self.consistency_metric}', "
                f"store_clusterers={self.store_clusterers}, "
                f"fitted={self.is_fitted_})")