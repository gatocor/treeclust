Utils (:mod:`treeclust.utils`)
=============================

.. automodule:: treeclust.utils

The utils module provides utility functions for clustering analysis, graph manipulation,
and data processing tasks commonly used throughout treeclust.

Core Utility Functions
-----------------------

assign_consistently
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: treeclust.utils.assign_consistently

   Assigns cluster labels consistently based on a reference using the Hungarian algorithm.
   
   **Key Features:**
   
   - Resolves label permutation issues between clusterings
   - Uses Hungarian algorithm for optimal assignment
   - Essential for comparing clustering results across runs
   - Maintains cluster meaning across multiple iterations

   **Example:**
   
   .. code-block:: python
   
      from treeclust.utils import assign_consistently
      import numpy as np
      
      # Reference clustering
      ref_labels = np.array([0, 0, 1, 1, 2, 2])
      
      # New clustering with different label assignment
      new_labels = np.array([2, 2, 0, 0, 1, 1])
      
      # Reassign consistently
      consistent_labels = assign_consistently(ref_labels, new_labels)
      # Result: [0, 0, 1, 1, 2, 2]

make_igraph
~~~~~~~~~~~

.. autofunction:: treeclust.utils.make_igraph

   Converts scipy sparse matrices to igraph Graph objects for network analysis.
   
   **Key Features:**
   
   - Efficient conversion from scipy sparse matrices
   - Preserves edge weights and graph structure  
   - Enables use of igraph algorithms and analysis tools
   - Essential for advanced network computations

   **Example:**
   
   .. code-block:: python
   
      from treeclust.utils import make_igraph
      from treeclust.neighbors import KNeighbors
      import numpy as np
      
      # Create adjacency matrix
      X = np.random.randn(100, 10)
      knn = KNeighbors(n_neighbors=5, mode='connectivity')
      adjacency = knn.fit_transform(X)
      
      # Convert to igraph
      graph = make_igraph(adjacency)
      
      # Use igraph functions
      modularity = graph.modularity(clusters)
      betweenness = graph.betweenness()

get_membership_
~~~~~~~~~~~~~~~

.. autofunction:: treeclust.utils.get_membership_

   Converts cluster labels to membership arrays for various algorithms.
   
   **Key Features:**
   
   - Standardizes cluster label formats across different algorithms
   - Handles different input types (numpy arrays, lists)
   - Ensures compatibility with downstream analysis tools
   - Manages edge cases like single clusters or noise labels

   **Example:**
   
   .. code-block:: python
   
      from treeclust.utils import get_membership_
      import numpy as np
      
      # Cluster labels  
      labels = np.array([0, 0, 1, 1, 2, 2, -1])  # -1 is noise
      
      # Get membership array
      membership = get_membership_(labels)

consensus_graph
~~~~~~~~~~~~~~~

.. autofunction:: treeclust.utils.consensus_graph

   Creates consensus graphs from multiple adjacency matrices or clustering results.
   
   **Key Features:**
   
   - Combines multiple graph representations for robustness
   - Supports different consensus strategies (voting, averaging, etc.)
   - Improves stability of downstream clustering
   - Essential for bootstrap and ensemble methods

   **Example:**
   
   .. code-block:: python
   
      from treeclust.utils import consensus_graph
      import numpy as np
      
      # Multiple adjacency matrices from bootstrap samples
      adjacency_list = [adj1, adj2, adj3, adj4, adj5]
      
      # Create consensus graph
      consensus_adj = consensus_graph(
          adjacency_list,
          consensus_threshold=0.6,
          method='voting'
      )

Graph and Network Functions
---------------------------

The utils module provides essential functions for graph manipulation and network analysis
that support the core clustering algorithms:

**Graph Conversion**:
   - ``make_igraph()``: Convert scipy sparse matrices to igraph objects
   - Format conversion between different graph representations

**Consensus Methods**:  
   - ``consensus_graph()``: Combine multiple graphs for robustness
   - Support for voting, averaging, and threshold-based consensus

**Label Management**:
   - ``assign_consistently()``: Resolve label permutation issues
   - ``get_membership_()``: Standardize cluster label formats

Usage Guidelines  
----------------

**When to Use These Functions**

- **assign_consistently()**: When comparing clusterings across multiple runs or algorithms
- **make_igraph()**: When you need igraph's advanced network analysis capabilities  
- **get_membership_()**: When preparing labels for specific algorithms or analysis tools
- **consensus_graph()**: When combining multiple graph representations for stability

**Integration Examples**

.. code-block:: python

   from treeclust.utils import assign_consistently, make_igraph
   from treeclust.clustering import Leiden
   from treeclust.neighbors import KNeighbors
   import numpy as np
   
   # Multiple clustering runs
   X = np.random.randn(200, 15)
   knn = KNeighbors(n_neighbors=10, mode='connectivity')  
   adjacency = knn.fit_transform(X)
   
   # Run clustering multiple times
   leiden = Leiden(resolution=1.0)
   ref_labels = leiden.fit_predict(adjacency)
   
   all_labels = []
   for i in range(5):
       leiden_run = Leiden(resolution=1.0, random_state=i)
       labels = leiden_run.fit_predict(adjacency)
       consistent_labels = assign_consistently(ref_labels, labels) 
       all_labels.append(consistent_labels)
   
   # Convert to igraph for advanced analysis
   graph = make_igraph(adjacency)
   community_structure = graph.community_leiden(resolution_parameter=1.0)

**Performance Tips**

- Use ``make_igraph()`` when you need multiple igraph operations on the same graph
- Cache igraph objects if performing repeated network analysis
- ``assign_consistently()`` works best with moderate numbers of clusters (< 100)
- For large graphs, consider sparse matrix operations before conversion