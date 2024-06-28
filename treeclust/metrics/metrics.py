from sknetwork.linalg.normalizer import normalize
from scipy.sparse import spmatrix
import numpy as np
from typing import Union, Any
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

def silhouette_scoring(X: Union[np.ndarray, spmatrix], lab: list) -> list:
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
    score = silhouette_score(X, lab)
    
    return score

def calinksi_harabasz_scoring(X: Union[np.ndarray, spmatrix], lab: list) -> list:
    """
    Calculate the calinski harabasz score for given data and number of clusters globally and per computed cluster.
    
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
    score = calinski_harabasz_score(X, lab)
    
    return score

def davies_bouldin_scoring(X: Union[np.ndarray, spmatrix], lab: list) -> list:
    """
    Calculate the davies bouldin score for given data and number of clusters globally and per computed cluster.
    
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
    score = davies_bouldin_score(X, lab)
    
    return score