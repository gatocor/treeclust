from sknetwork.linalg.normalizer import normalize
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