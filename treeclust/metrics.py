from sknetwork.linalg.normalizer import normalize
from sklearn.metrics import silhouette_score
from scipy.sparse import spmatrix
import numpy as np
from typing import Union, Any

def connectivity_probability(v: Union[np.ndarray, spmatrix], obj: Any) -> np.ndarray:
    """
    Calculates the connectivity probability for a vector or a matrix.

    Parameters:
    -----------
    v : Union[np.ndarray, spmatrix]
        Vector or matrix for which the connectivity probability is calculated.
    obj : Any
        Object containing the connectivity matrix.

    Returns:
    --------
    np.ndarray
        Normalized connectivity probability vector or matrix.
    """

    return normalize(obj.connectivity_matrix.dot(v))


def compute_silhouette(X: Union[np.ndarray, spmatrix], lab: list) -> list:
    """
    Calculate the silhouette score for given data and number of clusters globally and per computed cluster.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input samples (n_samples).
    lab : list
        The assigned clusters per cell
    
    Returns:
    float
        Global and per sample Silhouette score.
    """
    
    # Calculate Silhouette Score
    return [silhouette_score(X, lab)]



