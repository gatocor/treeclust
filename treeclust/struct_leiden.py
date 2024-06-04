from .struct_base import RobustClustering
import leidenalg
from .metrics import *
from .utils import *
import numpy as np
from typing import Optional, List, Dict, Callable, Any, Type

class RobustClusteringLeiden(RobustClustering):
    """
    A class for robust clustering using the Leiden algorithm.

    Attributes:
    ----------
    connectivity_matrix : np.ndarray
        The connectivity matrix used to create partitions.
    X : Optional[np.ndarray]
        Feature matrix, default is None.
    parameter_range : List[float]
        Range of parameters to explore for clustering.
    random_state : int
        Random seed for reproducibility.
    n_iter : int
        Number of iterations for the clustering algorithm.
    stop_criterion : str
        Criterion to stop the clustering process.
    stop_threshold : float
        Threshold for the stopping criterion.
    probability_metric : str
        Metric to measure the probability for stopping criterion.
    additional_metrics : Dict[str, Callable[[np.ndarray], float]]
        Additional metrics for evaluating clusters.
    marker_genes_matrix : Optional[np.ndarray]
        Matrix of marker genes, default is None.
    annotation_dict : Optional[Dict[str, Any]]
        Dictionary for annotations, default is None.
    partition : Type
        The partition type used by the Leiden algorithm.

    Methods:
    -------
    _find_clustering(resolution: float, seed: int) -> List[int]
        Finds the clustering for a given resolution and seed.
    """


    partition = None
    
    def __init__(
        self, 
        connectivity_matrix: np.ndarray, 
        X: Optional[np.ndarray] = None,
        parameter_range: np.ndarray = np.arange(0, 10, 0.1), 
        random_state: int = 0, 
        n_iter: int = 10, 
        stop_criterion: str = "connectivity_probability",
        stop_threshold: float = 0.5, 
        probability_metric: str = "connectivity_probability", 
        additional_metrics: Optional[Dict[str, Callable]] = {
            "connectivity_probability" : connectivity_probability
        },
        marker_genes_matrix: Optional[np.ndarray] = None,
        annotation_dict: Optional[Dict[str, Any]] = None,
        partition: Type = leidenalg.CPMVertexPartition
    ) -> None:
        """
        Initializes the RobustClusteringLeiden object with the given parameters.

        Parameters:
        ----------
        connectivity_matrix : np.ndarray
            The connectivity matrix used to create partitions.
        X : Optional[np.ndarray]
            Feature matrix, default is None.
        parameter_range : np.ndarray, optional
            Range of parameters to explore for clustering, default is np.arange(0, 10, 0.1).
        random_state : int, optional
            Random seed for reproducibility, default is 0.
        n_iter : int, optional
            Number of iterations for the clustering algorithm, default is 10.
        stop_criterion : str, optional
            Criterion to stop the clustering process, default is "connectivity_probability".
        stop_threshold : float, optional
            Threshold for the stopping criterion, default is 0.5.
        probability_metric : str, optional
            Metric to measure the probability for stopping criterion, default is "connectivity_probability".
        additional_metrics : Optional[Dict[str, Callable[[np.ndarray], float]]], optional
            Additional metrics for evaluating clusters, default is a dictionary with connectivity_probability function.
        marker_genes_matrix : Optional[np.ndarray], optional
            Matrix of marker genes, default is None.
        annotation_dict : Optional[Dict[str, Any]], optional
            Dictionary for annotations, default is None.
        partition : Type, optional
            The partition type used by the Leiden algorithm, default is leidenalg.CPMVertexPartition.
        """

        #General parameters shared by all metrics
        self.X = X
        self.connectivity_matrix = connectivity_matrix
        self.parameter_range = list(parameter_range)
        self.random_state = random_state
        self.n_iter = n_iter
        self.stop_criterion = stop_criterion
        self.stop_threshold = stop_threshold
        self.additional_metrics = additional_metrics
        self.marker_genes_matrix = marker_genes_matrix
        self.annotation_dict = annotation_dict
        self.probability_metric = probability_metric

        #Object sent to _find_clustering to make partitions (e.g X, connectivity_matrix, ...)
        self._clustering_object = make_igraph(self.connectivity_matrix)
        
        #Custom elements specific of this object
        self.partition = partition

    def _find_clustering(self, resolution: float, seed: int) -> List[int]:
        """
        Finds the clustering for a given resolution and seed.

        Parameters:
        ----------
        resolution : float
            The resolution parameter for the Leiden algorithm.
        seed : int
            The random seed for the Leiden algorithm.

        Returns:
        -------
        list
            The membership list of clusters.
        """

        return leidenalg.find_partition(self._clustering_object, self.partition, resolution_parameter=resolution, seed=seed).membership
