import numpy as np
import igraph
import warnings
from typing import Optional, List, Union, Any
from scipy import sparse
import scipy.sparse as sp

# Try to import cuRapids for GPU acceleration
try:
    import cugraph
    from cugraph import louvain as curapids_louvain
    import cuml.neighbors
    CURAPIDS_AVAILABLE = True
except ImportError:
    CURAPIDS_AVAILABLE = False

class Louvain:
    """
    Simplified Louvain clustering class for single parameter combinations.
    
    This class provides a clean sklearn-style interface for Louvain clustering
    with a single resolution, random state. For parameter exploration, use 
    ClusteringClassBootstrapper. For data bootstrapping, use the separate DataBootstrapper class.
    
    Key Features:
    - Single parameter clustering (use ClusteringClassBootstrapper for multiple)
    - Requires adjacency matrix input (no internal connectivity computation)
    - Clean sklearn-style fit/predict interface
    - Support for both dense and sparse adjacency matrices
    - Full GPU acceleration support via cuRapids (cuGraph)
    """

    def __init__(
        self,
        resolution: float = 1.0,
        random_state: int = 0,
        flavor: str = 'auto',
        n_repetitions: int = 10,
        consistency_metric: str = 'ari'
    ):
        """
        Initialize the Louvain clustering class.

        Parameters:
        -----------
        resolution : float, default=1.0
            Resolution parameter for Louvain clustering.
            Higher values lead to more clusters.
            
        random_state : int, default=0
            Random seed for reproducible clustering.
            
        flavor : str, default='auto'
            Louvain algorithm implementation to use. Options:
            - 'auto': Use cuRapids if available, fallback to igraph
            - 'curapids': Use cuRapids GPU implementation (requires cugraph)
            - 'igraph': Use igraph CPU implementation
            
        n_repetitions : int, default=10
            Number of repetitions to run for consistency checking.
            If > 1, multiple runs are performed and consistency metrics are computed.
            
        consistency_metric : str, default='ari'
            Metric to use for measuring consistency between repetitions. Options:
            - 'ari': Adjusted Rand Index (range: -1 to 1, higher is better)
            - 'ami': Adjusted Mutual Information (range: 0 to 1, higher is better)
            - 'nmi': Normalized Mutual Information (range: 0 to 1, higher is better)
            - 'homogeneity': Homogeneity score (range: 0 to 1, higher is better)
            - 'completeness': Completeness score (range: 0 to 1, higher is better)
            - 'v_measure': V-measure (range: 0 to 1, higher is better)
        """
        self.resolution = float(resolution)
        self.random_state = int(random_state)
        self.flavor = flavor
        self.n_repetitions = int(n_repetitions)
        self.consistency_metric = consistency_metric
        
        # Validate consistency metric
        valid_metrics = ['ari', 'ami', 'nmi', 'homogeneity', 'completeness', 'v_measure']
        if self.consistency_metric not in valid_metrics:
            raise ValueError(f"Unknown consistency metric: {self.consistency_metric}. "
                           f"Available options: {valid_metrics}")
        
        # Validate n_repetitions
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be >= 1")
        
        # Determine which Louvain implementation to use
        if self.flavor == 'auto':
            self.use_curapids = CURAPIDS_AVAILABLE
        elif self.flavor == 'curapids':
            if not CURAPIDS_AVAILABLE:
                raise ImportError(
                    "cuRapids (cugraph) is not available. "
                    "Install with: conda install -c rapidsai cugraph, "
                    "or use flavor='igraph' for CPU implementation."
                )
            self.use_curapids = True
        elif self.flavor == 'igraph':
            self.use_curapids = False
        else:
            raise ValueError(f"Unknown flavor: {self.flavor}. "
                           f"Available options: ['auto', 'curapids', 'igraph']")
                
        # Results storage
        self.labels_ = None
        self.adjacency_ = None
        self.is_fitted_ = False
        
        # Consistency checking storage
        self.all_labels_ = []  # Store all repetition results
        self.consistency_scores_ = []  # Store pairwise consistency scores
        self.mean_consistency_ = None  # Mean consistency score
        self.std_consistency_ = None   # Standard deviation of consistency scores
        self.per_cluster_consistency_ = None  # Per-cluster consistency information
        self.cell_confidence_ = None  # Per-cell confidence scores across repetitions
        
    def fit(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> 'Louvain':
        """
        Fit the Louvain clustering algorithm to the adjacency matrix.
        
        Parameters:
        -----------
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Adjacency matrix of shape (n_samples, n_samples).
            Can be dense numpy array or scipy sparse matrix.
            
        Returns:
        --------
        self : Louvain
            Returns self for method chaining.
        """
        # Store the adjacency matrix
        self.adjacency_ = adjacency_matrix.copy()
        
        # Clear previous results
        self.all_labels_ = []
        self.consistency_scores_ = []
        
        # Run clustering for specified number of repetitions
        for i in range(self.n_repetitions):
            # Use different random seed for each repetition
            current_seed = self.random_state + i
            
            # Set random seed
            np.random.seed(current_seed)
            
            # Run single clustering iteration
            if self.use_curapids:
                labels = self._fit_single_curapids(current_seed)
            else:
                labels = self._fit_single_igraph(current_seed)
            
            self.all_labels_.append(labels)
        
        # Use the first run as the main result
        self.labels_ = self.all_labels_[0]
        
        # Compute consistency scores if multiple repetitions
        if self.n_repetitions > 1:
            self._compute_consistency_scores()
            self.cell_confidence_ = self._compute_cell_confidence()
        else:
            self.cell_confidence_ = None
        
        self.is_fitted_ = True
        return self
        
    def _fit_single_curapids(self, random_seed: int):
        """Fit using cuRapids GPU implementation for a single run."""
        try:
            import cudf
            import cugraph
            
            # Create edge list from adjacency matrix (avoid dense conversion)
            if sparse.issparse(self.adjacency_):
                # Convert sparse matrix to COO format for edge list creation
                self.adjacency_ = self.adjacency_.tocoo()
                
                # Get edges (keep all edges for undirected graph)
                sources = self.adjacency_.row
                destinations = self.adjacency_.col
                weights = self.adjacency_.data
            else:
                # Convert dense matrix to edge list
                rows, cols = np.nonzero(self.adjacency_)
                sources = rows
                destinations = cols
                weights = self.adjacency_[rows, cols]
            
            # Create cuDF DataFrame for edges
            edges_df = cudf.DataFrame({
                'src': sources,
                'dst': destinations,
                'weight': weights
            })
            
            # Create cuGraph
            G = cugraph.Graph()
            G.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight')
                                                    
            # Run Louvain clustering
            result_df = cugraph.louvain(
                G,
                resolution=self.resolution,
                random_state=random_seed
            )
            
            # Extract labels and convert to CPU
            labels_cudf = result_df['partition'].to_pandas()
            vertex_ids = result_df['vertex'].to_pandas()
            
            # Create labels array in original order
            labels = [0] * len(vertex_ids)
            for vertex_id, label in zip(vertex_ids, labels_cudf):
                labels[vertex_id] = label
            
            return labels
                    
        except ImportError as e:
            raise ImportError(f"cuRapids dependencies missing: {e}. "
                            f"Install with: conda install -c rapidsai cugraph cudf")
        
    def _fit_single_igraph(self, random_seed: int):
        """Fit using igraph CPU implementation for a single run."""
        # Create igraph using appropriate method
        if sparse.issparse(self.adjacency_):
            # Convert sparse matrix to edge list (avoids dense conversion)
            self.adjacency_ = self.adjacency_.tocoo()  # Convert to COO format
            
            # Create edge list from COO sparse matrix
            edges = list(zip(self.adjacency_.row, self.adjacency_.col))
            weights = self.adjacency_.data.tolist()
            
            # Create igraph from edge list
            G = igraph.Graph(
                n=self.adjacency_.shape[0],
                edges=edges,
                directed=False
            )
            
            # Add weights if they exist and are not all 1
            if not np.allclose(weights, 1.0):
                G.es['weight'] = weights
                
        else:
            # Use Adjacency for dense matrices
            G = igraph.Graph.Adjacency(
                self.adjacency_.tolist(),
                mode='undirected'
            )
                
        # Run Louvain clustering with igraph
        # igraph's community_multilevel is the Louvain algorithm
        # Note: igraph doesn't directly support random_state in community_multilevel,
        # but we've already set np.random.seed in the fit method
        community = G.community_multilevel(
            weights=None if not hasattr(G.es, 'weight') else G.es['weight'],
            resolution=self.resolution,
            return_levels=False
        )
        
        # Return results
        return community.membership
    
    def _compute_consistency_scores(self):
        """Compute pairwise consistency scores between all repetitions."""
        try:
            from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, 
                                       normalized_mutual_info_score, homogeneity_score,
                                       completeness_score, v_measure_score)
            
            # Map metric names to functions
            metric_functions = {
                'ari': adjusted_rand_score,
                'ami': adjusted_mutual_info_score,
                'nmi': normalized_mutual_info_score,
                'homogeneity': homogeneity_score,
                'completeness': completeness_score,
                'v_measure': v_measure_score
            }
            
            metric_func = metric_functions[self.consistency_metric]
            
            # Compute all pairwise scores
            n_reps = len(self.all_labels_)
            scores = []
            
            for i in range(n_reps):
                for j in range(i + 1, n_reps):
                    score = metric_func(self.all_labels_[i], self.all_labels_[j])
                    scores.append(score)
            
            self.consistency_scores_ = scores
            
            if scores:
                self.mean_consistency_ = np.mean(scores)
                self.std_consistency_ = np.std(scores) if len(scores) > 1 else 0.0
            else:
                self.mean_consistency_ = None
                self.std_consistency_ = None
                
        except ImportError:
            warnings.warn("sklearn not available. Consistency scores will not be computed.")
            self.consistency_scores_ = []
            self.mean_consistency_ = None
            self.std_consistency_ = None
    
    def get_consistency_summary(self) -> dict:
        """
        Get summary of consistency across repetitions.
        
        Returns:
        --------
        summary : dict
            Dictionary containing consistency statistics:
            - 'n_repetitions': Number of repetitions performed
            - 'consistency_metric': Metric used for consistency measurement
            - 'mean_consistency': Mean consistency score
            - 'std_consistency': Standard deviation of consistency scores
            - 'all_scores': List of all pairwise consistency scores
            - 'all_labels': List of label arrays from all repetitions
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return {
            'n_repetitions': self.n_repetitions,
            'consistency_metric': self.consistency_metric,
            'mean_consistency': self.mean_consistency_,
            'std_consistency': self.std_consistency_,
            'all_scores': self.consistency_scores_,
            'all_labels': self.all_labels_
        }
        
    def fit_predict(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> List[int]:
        """
        Fit the model and predict cluster labels.
        
        Parameters:
        -----------
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Adjacency matrix of shape (n_samples, n_samples) representing the graph structure.
            
        Returns:
        --------
        labels : List[int]
            Cluster labels for each sample.
        """
        self.fit(adjacency_matrix)
        return self.labels_
    
    def _compute_cell_confidence(self) -> np.ndarray:
        """
        Compute confidence score for each cell based on consistency across repetitions.
        
        The confidence score for each cell is the fraction of repetitions where
        the cell was assigned to the same cluster as in the majority assignment,
        after matching labels between repetitions using the Hungarian algorithm.
        
        Returns:
        --------
        cell_confidence : np.ndarray
            Array of confidence scores (0.0 to 1.0) for each cell.
            Higher values indicate more consistent cluster assignments.
        """
        if len(self.all_labels_) < 2:
            return None
            
        from .utils import assign_consistently
            
        n_cells = len(self.all_labels_[0])
        n_repetitions = len(self.all_labels_)
        
        # Use the first repetition as reference
        reference_labels = self.all_labels_[0]
        
        # Match all other repetitions to the reference
        matched_labels = [reference_labels]  # First repetition is the reference
        
        for rep_idx in range(1, n_repetitions):
            # Match this repetition's labels to the reference
            matched_rep_labels = assign_consistently(reference_labels, self.all_labels_[rep_idx])
            matched_labels.append(matched_rep_labels)
        
        # Now compute confidence for each cell
        confidence_scores = np.zeros(n_cells, dtype=float)
        
        for cell_idx in range(n_cells):
            # Get matched cluster assignments for this cell across all repetitions
            cell_assignments = [labels[cell_idx] for labels in matched_labels]
            
            # Count frequency of each cluster assignment
            from collections import Counter
            assignment_counts = Counter(cell_assignments)
            
            # Find the most frequent assignment and its count
            most_common_assignment, max_count = assignment_counts.most_common(1)[0]
            
            # Confidence is the fraction of repetitions with the most common assignment
            confidence_scores[cell_idx] = max_count / n_repetitions
        
        return confidence_scores
           
    def __repr__(self):
        """String representation of the Louvain object."""
        implementation = "cuRapids" if self.use_curapids else "igraph"
        
        if self.is_fitted_:
            n_clusters = len(set(self.labels_))
            base_repr = (f"Louvain(resolution={self.resolution}, "
                        f"random_state={self.random_state}, "
                        f"flavor='{self.flavor}', "
                        f"n_repetitions={self.n_repetitions}, "
                        f"consistency_metric='{self.consistency_metric}', "
                        f"implementation={implementation}, "
                        f"fitted=True, n_clusters={n_clusters}")
            
            if self.n_repetitions > 1 and self.mean_consistency_ is not None:
                consistency_str = f", consistency={self.mean_consistency_:.3f}±{self.std_consistency_:.3f}"
                return base_repr + consistency_str + ")"
            else:
                return base_repr + ")"
        else:
            return (f"Louvain(resolution={self.resolution}, "
                   f"random_state={self.random_state}, "
                   f"flavor='{self.flavor}', "
                   f"n_repetitions={self.n_repetitions}, "
                   f"consistency_metric='{self.consistency_metric}', "
                   f"implementation={implementation}, "
                   f"fitted=False)")