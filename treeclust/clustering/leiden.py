import leidenalg
import numpy as np
import igraph
import warnings
from typing import Optional, List, Union, Any
from scipy import sparse
import scipy.sparse as sp

# Import centralized availability flags
from .. import CURAPIDS_AVAILABLE

class Leiden:
    """
    Simplified Leiden clustering class for single parameter combinations.
    
    This class provides a clean sklearn-style interface for Leiden clustering
    with a single resolution, partition type, and random state. For parameter
    exploration, use ParameterBootstrapper. For data bootstrapping, use the
    separate DataBootstrapper class.
    
    Key Features:
    - Single parameter clustering (use ParameterBootstrapper for multiple)
    - Requires adjacency matrix input (no internal connectivity computation)
    - Clean sklearn-style fit/predict interface
    - Support for both dense and sparse adjacency matrices
    - Full GPU acceleration support via cuRapids (cuGraph)
    """

    def __init__(
        self,
        resolution: float = 1.0,
        random_state: int = 0,
        partition_type: str = 'RB',
        flavor: str = 'auto',
        n_repetitions: int = 10,
        consistency_metric: str = 'ari'
    ):
        """
        Initialize the Leiden clustering class.

        Parameters:
        -----------
        resolution : float, default=1.0
            Resolution parameter for Leiden clustering.
            
        random_state : int, default=0
            Random seed for reproducible clustering.
            
        partition_type : str, default='RB'
            Partition method to use. Options:
            - 'CPM': Constant Potts Model
            - 'modularity': Modularity optimization
            - 'RB': Reichardt and Bornholdt
            - 'RBER': Reichardt and Bornholdt with self-loops
            - 'surprise': Surprise optimization
            Note: When using cuRapids (flavor='curapids'), partition_type is ignored 
            as cuRapids only supports modularity-based partitioning.
            
        flavor : str, default='auto'
            Leiden algorithm implementation to use. Options:
            - 'auto': Use cuRapids if available, fallback to leidenalg
            - 'curapids': Use cuRapids GPU implementation (requires cugraph)
            - 'leidenalg': Use leidenalg CPU implementation
            Note: cuRapids only supports modularity-based partitioning, so partition_type
            is ignored when using cuRapids flavor.
            
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
        self.partition_type = partition_type
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
        
        # Determine which Leiden implementation to use
        if self.flavor == 'auto':
            self.use_curapids = CURAPIDS_AVAILABLE
        elif self.flavor == 'curapids':
            if not CURAPIDS_AVAILABLE:
                raise ImportError(
                    "cuRapids (cugraph) is not available. "
                    "Install with: conda install -c rapidsai cugraph, "
                    "or use flavor='leidenalg' for CPU implementation."
                )
            self.use_curapids = True
        elif self.flavor == 'leidenalg':
            self.use_curapids = False
        else:
            raise ValueError(f"Unknown flavor: {self.flavor}. "
                           f"Available options: ['auto', 'curapids', 'leidenalg']")
                
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
        
        # Partition type mapping for leidenalg
        self._partition_classes = {
            'CPM': leidenalg.CPMVertexPartition,
            'modularity': leidenalg.ModularityVertexPartition,
            'RB': leidenalg.RBConfigurationVertexPartition,
            'RBER': leidenalg.RBERVertexPartition,
            'surprise': leidenalg.SurpriseVertexPartition
        }
        
        # Validate partition type for leidenalg (cuRapids has more limited options)
        if not self.use_curapids and self.partition_type not in self._partition_classes:
            raise ValueError(f"Unknown partition type for leidenalg: {self.partition_type}. "
                           f"Available options: {list(self._partition_classes.keys())}")
        
        # cuRapids only supports modularity-based partitioning (partition_type ignored)
        if self.use_curapids and self.partition_type not in ['modularity', 'CPM']:
            warnings.warn(
                f"cuRapids only supports modularity-based partitioning. "
                f"partition_type='{self.partition_type}' will be ignored and "
                f"modularity-based partitioning will be used instead.",
                UserWarning
            )
        
    def fit(self, adjacency_matrix: Union[np.ndarray, sp.spmatrix]) -> 'Leiden':
        """
        Fit the Leiden clustering algorithm to the adjacency matrix.
        
        Parameters:
        -----------
        adjacency_matrix : np.ndarray or scipy.sparse matrix
            Adjacency matrix of shape (n_samples, n_samples).
            Can be dense numpy array or scipy sparse matrix.
            
        Returns:
        --------
        self : Leiden
            Returns self for method chaining.
        """
        # Store the adjacency matrix
        self.adjacency_ = adjacency_matrix.copy()
        
        # Clear previous results
        self.all_labels_ = []
        self.consistency_scores_ = []
        self.per_cluster_consistency_ = None
        
        # Run clustering for specified number of repetitions
        for i in range(self.n_repetitions):
            # Use different random seed for each repetition
            current_seed = self.random_state + i
            
            # Run single clustering iteration
            if self.use_curapids:
                labels = self._fit_single_curapids(current_seed)
            else:
                labels = self._fit_single_leidenalg(current_seed)
            
            self.all_labels_.append(labels)
        
        # Compute consistency scores if multiple repetitions
        if self.n_repetitions > 1:
            self._compute_consistency_scores()
            self.per_cluster_consistency_ = self._compute_per_cluster_consistency()
            # Create final labels that correspond to cluster track IDs
            self.labels_ = self._create_consistent_final_labels()
            self.cell_confidence_ = self._compute_cell_confidence()
        else:
            # Use the first run as the main result for single repetition
            self.labels_ = self.all_labels_[0]
            self.per_cluster_consistency_ = None
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
                                    
            # Run Leiden clustering
            # Note: cuRapids uses modularity-based partitioning regardless of partition_type
            result_df = cugraph.leiden(
                G,
                resolution=self.resolution,
                random_state=random_seed
            )
            
            # Extract labels and convert to CPU
            labels_cudf = result_df['partition'].to_pandas()
            vertex_ids = result_df['vertex'].to_pandas()
            
            # Create labels array in original order
            labels = [0] * len(vertex_ids)
            for i, (vertex_id, label) in enumerate(zip(vertex_ids, labels_cudf)):
                labels[vertex_id] = label
            
            return labels
                
        except ImportError as e:
            raise ImportError(f"cuRapids dependencies missing: {e}. "
                            f"Install with: conda install -c rapidsai cugraph cudf")
    
    def _fit_single_leidenalg(self, random_seed: int):
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
        
        # Get partition class
        partition_class = self._partition_classes[self.partition_type]
        
        # Prepare partition arguments
        partition_kwargs = {}
        
        # Only add resolution_parameter for partition types that support it
        if self.partition_type in ['CPM', 'RB', 'RBER']:
            partition_kwargs['resolution_parameter'] = self.resolution
        else:
            warnings.warn("Resolution parameter is not used for partition type "
                         f"'{self.partition_type}' and will be ignored.", UserWarning)
        
        # Create and optimize partition
        if hasattr(G.es, 'weight'):
            # Weighted graph
            partition = partition_class(
                G,
                weights=G.es['weight'],
                **partition_kwargs
            )
        else:
            # Unweighted graph
            partition = partition_class(G, **partition_kwargs)
        
        # Optimize partition
        optimiser = leidenalg.Optimiser()
        optimiser.set_rng_seed(random_seed)
        optimiser.optimise_partition(partition)
        
        # Return results
        return partition.membership
    
    def _create_consistent_final_labels(self):
        """
        Create final labels that use cluster track IDs to ensure correspondence
        with stability metrics.
        
        Returns:
        --------
        np.ndarray: Final cluster labels using track IDs
        """
        # Convert label arrays to cluster membership sets for each repetition
        rep_clusters = []
        for labels in self.all_labels_:
            clusters_in_rep = {}
            for cluster_id in np.unique(labels):
                members = set(np.where(labels == cluster_id)[0])
                clusters_in_rep[cluster_id] = members
            rep_clusters.append(clusters_in_rep)

        # Find cluster correspondences across repetitions using overlap matching
        cluster_tracks = self._match_clusters_across_repetitions(rep_clusters)
        
        # Create final labels using track IDs
        n_samples = len(self.all_labels_[0])
        final_labels = np.full(n_samples, -1, dtype=int)  # Initialize with -1 (unassigned)
        
        # Use the first repetition as the base, but assign track IDs
        first_rep_labels = self.all_labels_[0]
        first_rep_clusters = {}
        for cluster_id in np.unique(first_rep_labels):
            members = set(np.where(first_rep_labels == cluster_id)[0])
            first_rep_clusters[cluster_id] = members
        
        # For each track, find its appearance in the first repetition and assign track ID
        for track_id, track_info in cluster_tracks.items():
            # Find appearances in first repetition (rep_idx=0)
            first_rep_appearances = [
                cluster_id for rep_idx, cluster_id in track_info['appearances']
                if rep_idx == 0
            ]
            
            if first_rep_appearances:
                # This track appeared in first repetition - use its members
                original_cluster_id = first_rep_appearances[0]
                member_indices = list(first_rep_clusters[original_cluster_id])
                final_labels[member_indices] = track_id
            # Note: Tracks that didn't appear in first repetition are not included
            # in final labels, which is consistent with the original behavior
        
        # Handle any samples that weren't assigned (shouldn't happen normally)
        unassigned_mask = final_labels == -1
        if np.any(unassigned_mask):
            # Find the next available track ID
            max_track_id = max(cluster_tracks.keys()) if cluster_tracks else -1
            next_track_id = max_track_id + 1
            
            # Group unassigned samples by their original cluster in first repetition
            for cluster_id in np.unique(first_rep_labels):
                cluster_mask = first_rep_labels == cluster_id
                if np.any(cluster_mask & unassigned_mask):
                    final_labels[cluster_mask & unassigned_mask] = next_track_id
                    next_track_id += 1
        
        return final_labels
    
    def _compute_per_cluster_consistency(self):
        """
        Compute consistency scores for each cluster across repetitions.
        
        This method properly handles label permutations by matching clusters
        based on member overlap rather than label values.
        
        Returns:
        --------
        cluster_consistency : dict
            Dictionary with cluster identifiers as keys and consistency info as values.
            Each value contains: {'consistency': float, 'stability': float, 'size': int}
        """
        if len(self.all_labels_) < 2:
            return {}
        
        # Convert label arrays to cluster membership sets for each repetition
        rep_clusters = []
        for labels in self.all_labels_:
            clusters_in_rep = {}
            for cluster_id in np.unique(labels):
                members = set(np.where(labels == cluster_id)[0])
                clusters_in_rep[cluster_id] = members
            rep_clusters.append(clusters_in_rep)
        
        # Find cluster correspondences across repetitions using overlap matching
        cluster_tracks = self._match_clusters_across_repetitions(rep_clusters)
        
        # Compute consistency for each tracked cluster
        cluster_consistency = {}
        
        for track_id, track_info in cluster_tracks.items():
            consistency_scores = []
            sizes = []
            n_appearances = len(track_info['appearances'])
            
            # Get all member sets for this tracked cluster
            member_sets = []
            for rep_idx, cluster_id in track_info['appearances']:
                member_sets.append(rep_clusters[rep_idx][cluster_id])
                sizes.append(len(rep_clusters[rep_idx][cluster_id]))
            
            # Compute pairwise Jaccard similarities
            for i in range(len(member_sets)):
                for j in range(i + 1, len(member_sets)):
                    set_i, set_j = member_sets[i], member_sets[j]
                    if len(set_i) > 0 or len(set_j) > 0:
                        intersection = len(set_i.intersection(set_j))
                        union = len(set_i.union(set_j))
                        jaccard = intersection / union if union > 0 else 0.0
                        consistency_scores.append(jaccard)
            
            # Calculate statistics
            mean_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            std_consistency = np.std(consistency_scores) if len(consistency_scores) > 1 else 0.0
            stability = n_appearances / len(self.all_labels_)
            mean_size = np.mean(sizes) if sizes else 0.0
            std_size = np.std(sizes) if len(sizes) > 1 else 0.0
            
            cluster_consistency[track_id] = {
                'mean_consistency': mean_consistency,
                'std_consistency': std_consistency,
                'stability': stability,
                'mean_size': mean_size,
                'std_size': std_size,
                'occurrences': n_appearances,
                'consistency_scores': consistency_scores,
                'sizes': sizes
            }
        
        return cluster_consistency
    
    def _match_clusters_across_repetitions(self, rep_clusters):
        """
        Match clusters across repetitions based on member overlap.
        
        Uses a greedy matching approach: for each cluster in each repetition,
        find the best match in other repetitions based on Jaccard similarity.
        
        Parameters:
        -----------
        rep_clusters : list of dict
            List where each element is a dict {cluster_id: member_set} for one repetition
            
        Returns:
        --------
        cluster_tracks : dict
            Dictionary where each key is a track_id and value contains:
            {'appearances': [(rep_idx, cluster_id), ...], 'representative_members': set}
        """
        cluster_tracks = {}
        next_track_id = 0
        
        # For each repetition, try to match its clusters to existing tracks
        for rep_idx, clusters_in_rep in enumerate(rep_clusters):
            unmatched_clusters = set(clusters_in_rep.keys())
            
            # Try to match each cluster to an existing track
            for cluster_id in list(unmatched_clusters):
                cluster_members = clusters_in_rep[cluster_id]
                best_track = None
                best_jaccard = 0.0
                
                # Find best matching existing track
                for track_id, track_info in cluster_tracks.items():
                    # Compare with representative members of this track
                    repr_members = track_info['representative_members']
                    
                    if len(cluster_members) > 0 or len(repr_members) > 0:
                        intersection = len(cluster_members.intersection(repr_members))
                        union = len(cluster_members.union(repr_members))
                        jaccard = intersection / union if union > 0 else 0.0
                        
                        if jaccard > best_jaccard and jaccard > 0.1:  # Minimum threshold
                            best_jaccard = jaccard
                            best_track = track_id
                
                # Assign cluster to best track or create new track
                if best_track is not None:
                    cluster_tracks[best_track]['appearances'].append((rep_idx, cluster_id))
                    # Update representative members (union of all members seen so far)
                    cluster_tracks[best_track]['representative_members'].update(cluster_members)
                else:
                    # Create new track
                    cluster_tracks[next_track_id] = {
                        'appearances': [(rep_idx, cluster_id)],
                        'representative_members': cluster_members.copy()
                    }
                    next_track_id += 1
                
                unmatched_clusters.remove(cluster_id)
        
        return cluster_tracks

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
            - 'mean_consistency': Mean global consistency score
            - 'std_consistency': Standard deviation of global consistency scores
            - 'all_scores': List of all pairwise global consistency scores
            - 'all_labels': List of label arrays from all repetitions
            - 'per_cluster_consistency': Dict with per-cluster consistency information
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return {
            'n_repetitions': self.n_repetitions,
            'consistency_metric': self.consistency_metric,
            'mean_consistency': self.mean_consistency_,
            'std_consistency': self.std_consistency_,
            'all_scores': self.consistency_scores_,
            'all_labels': self.all_labels_,
            'per_cluster_consistency': self.per_cluster_consistency_
        }
        
    def get_per_cluster_consistency_summary(self) -> dict:
        """
        Get a summary of per-cluster consistency metrics.
        
        Returns:
        --------
        summary : dict
            Dictionary containing per-cluster consistency statistics:
            - 'n_clusters': Total number of unique clusters across all repetitions
            - 'cluster_stats': Dict with cluster ID as key and stats as values
            - 'most_consistent_clusters': List of (cluster_id, consistency) sorted by consistency
            - 'most_stable_clusters': List of (cluster_id, stability) sorted by stability
            - 'largest_clusters': List of (cluster_id, mean_size) sorted by size
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        if self.per_cluster_consistency_ is None:
            return {"message": "Per-cluster consistency not available (n_repetitions must be > 1)"}
        
        cluster_stats = {}
        consistencies = []
        stabilities = []
        sizes = []
        
        for cluster_id, info in self.per_cluster_consistency_.items():
            stats = {
                'mean_consistency': info['mean_consistency'],
                'std_consistency': info['std_consistency'], 
                'stability': info['stability'],
                'mean_size': info['mean_size'],
                'std_size': info['std_size'],
                'occurrences': info['occurrences']
            }
            cluster_stats[cluster_id] = stats
            
            consistencies.append((cluster_id, info['mean_consistency']))
            stabilities.append((cluster_id, info['stability']))
            sizes.append((cluster_id, info['mean_size']))
        
        # Sort by different metrics
        most_consistent = sorted(consistencies, key=lambda x: x[1], reverse=True)[:10]
        most_stable = sorted(stabilities, key=lambda x: x[1], reverse=True)[:10]  
        largest = sorted(sizes, key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'n_clusters': len(cluster_stats),
            'cluster_stats': cluster_stats,
            'most_consistent_clusters': most_consistent,
            'most_stable_clusters': most_stable,
            'largest_clusters': largest
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
        """String representation of the Leiden object."""
        implementation = "cuRapids" if self.use_curapids else "leidenalg"
        
        if self.is_fitted_:
            n_clusters = len(set(self.labels_))
            base_repr = (f"Leiden(resolution={self.resolution}, "
                        f"random_state={self.random_state}, "
                        f"partition_type='{self.partition_type}', "
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
            return (f"Leiden(resolution={self.resolution}, "
                   f"random_state={self.random_state}, "
                   f"partition_type='{self.partition_type}', "
                   f"flavor='{self.flavor}', "
                   f"n_repetitions={self.n_repetitions}, "
                   f"consistency_metric='{self.consistency_metric}', "
                   f"implementation={implementation}, "
                   f"fitted=False)")