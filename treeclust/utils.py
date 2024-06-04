import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, spmatrix, issparse
import igraph
from typing import List, Dict, Any, Union

def assign_consistently(ref: Union[np.ndarray, List[int]], v: Union[np.ndarray, List[int]]) -> List[int]:
    """
    Assigns values consistently based on the reference using the Hungarian algorithm.

    Parameters:
    ----------
    ref : Union[np.ndarray, List[int]]
        Reference array.
    v : Union[np.ndarray, List[int]]
        Values to reassign based on the reference.

    Returns:
    -------
    List[int]
        List of reassigned values.
    """

    df = pd.DataFrame()
    df["ref"] = ref
    df["reassign"] = v
    df["count"] = 1
    df = df.groupby(by=["ref","reassign"]).count().unstack().fillna(0)
    df = (df.div(df.sum(axis=1),axis=0) + df.div(df.sum(axis=0),axis=1))/2

    while df.shape[0] != df.shape[1]:
        if df.shape[0] > df.shape[1]:
    
            df[df.shape[1]] = 0
    
        elif df.shape[0] < df.shape[1]:
            
            df.loc[df.shape[0],:] = 0
    
    m = linear_sum_assignment(df.values, maximize=True)
    map = {j:i for i,j in zip(m[0],m[1])}
        
    return [map[i] for i in v]

def make_igraph(adjacency: spmatrix) -> igraph.Graph:
    """
    Creates an igraph object from an adjacency matrix.

    Parameters:
    ----------
    adjacency : spmatrix
        Adjacency matrix in any sparse format from SciPy.

    Returns:
    -------
    igraph.Graph
        An igraph object representing the graph.
    """

    if not issparse(adjacency):
        A = coo_matrix(adjacency)
    else:
        A = adjacency.tocoo()
    edges = [(i,j) for i,j in zip(A.col,A.row)]
    edges_attrs = {"weights" : A.data}

    g = igraph.Graph(edges = edges, edge_attrs = edges_attrs)

    return g

def get_membership_(clusters: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Gets the membership matrix from a membership array or list.

    Parameters:
    ----------
    clusters : Union[np.ndarray, List[int]]
        Membership array or list.

    Returns:
    -------
    np.ndarray
        Membership matrix.
    """

    l = len(np.unique(clusters))
    matrix = np.zeros([len(clusters),l])

    for i,j in enumerate(clusters):
        matrix[i,j] = 1

    return matrix