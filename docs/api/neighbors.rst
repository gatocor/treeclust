Neighbors (:mod:`treeclust.neighbors`)
=====================================

.. automodule:: treeclust.neighbors

The neighbors module provides algorithms for computing k-nearest neighbors and shared neighbors 
with sklearn compatibility, consensus methods, and bootstrapping capabilities.

Core Neighbors Classes
----------------------

KNeighbors
~~~~~~~~~~

.. autoclass:: treeclust.neighbors.KNeighbors
   :members:
   :inherited-members:
   :show-inheritance:

   Efficient k-nearest neighbors implementation with sklearn compatibility.
   
   **Key Features:**
   
   - sklearn-style interface for easy integration
   - Support for different distance metrics and algorithms
   - Flexible output modes (connectivity, distance, indices)
   - Optional GPU acceleration support

   **Example:**
   
   .. code-block:: python
   
      from treeclust.neighbors import KNeighbors
      import numpy as np
      
      # Create sample data
      X = np.random.randn(1000, 50)
      
      # Compute k-nearest neighbors
      knn = KNeighbors(n_neighbors=15, metric='euclidean')
      
      # Get connectivity matrix
      adjacency = knn.fit_transform(X, mode='connectivity')
      
      # Get distances
      distances = knn.fit_transform(X, mode='distance')

MutualNearestNeighbors
~~~~~~~~~~~~~~~

.. autoclass:: treeclust.neighbors.MutualNearestNeighbors
   :members:
   :inherited-members:
   :show-inheritance:

   Shared neighbors computation for improved clustering in high-dimensional spaces.
   
   **Key Features:**
   
   - Jaccard and other similarity measures
   - Reduces curse of dimensionality effects  
   - Improved clustering quality in high dimensions
   - sklearn-compatible interface

   **Example:**
   
   .. code-block:: python
   
      from treeclust.neighbors import MutualNearestNeighbors
      import numpy as np
      
      # High-dimensional data
      X = np.random.randn(500, 100)
      
      # Compute shared neighbors
      sn = MutualNearestNeighbors(
          n_neighbors=20,
          similarity='jaccard',
          min_shared=3
      )
      
      # Get shared neighbors matrix
      shared_matrix = sn.fit_transform(X)

Consensus Methods
-----------------

ConsensusNearestNeighbors
~~~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.neighbors.ConsensusNearestNeighbors
   :members:
   :inherited-members:
   :show-inheritance:

   Consensus k-nearest neighbors using multiple parameter sets or bootstrap sampling.
   
   **Key Features:**
   
   - Combines multiple k-NN computations for robustness
   - Bootstrap sampling for stability
   - Parameter consensus across multiple settings
   - Improved neighbor quality through averaging

   **Example:**
   
   .. code-block:: python
   
      from treeclust.neighbors import ConsensusNearestNeighbors
      import numpy as np
      
      # Create consensus k-NN
      consensus_knn = ConsensusNearestNeighbors(
          n_neighbors_list=[10, 15, 20],
          n_bootstrap=50,
          consensus_threshold=0.5
      )
      
      # Fit and get consensus adjacency
      X = np.random.randn(300, 40)
      consensus_adj = consensus_knn.fit_transform(X)

ConsensusMutualNearestNeighbors  
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.neighbors.ConsensusMutualNearestNeighbors
   :members:
   :inherited-members:
   :show-inheritance:

   Consensus shared neighbors combining multiple shared neighbor computations.
   
   **Key Features:**
   
   - Consensus across multiple shared neighbor parameters
   - Bootstrap sampling for improved stability
   - Robust neighbor identification in high dimensions
   - Enhanced clustering performance

   **Example:**
   
   .. code-block:: python
   
      from treeclust.neighbors import ConsensusMutualNearestNeighbors
      import numpy as np
      
      # Create consensus shared neighbors
      consensus_sn = ConsensusMutualNearestNeighbors(
          n_neighbors_list=[15, 20, 25], 
          similarity='jaccard',
          n_bootstrap=30,
          min_shared=2
      )
      
      # Fit and get consensus matrix
      X = np.random.randn(400, 80)
      consensus_matrix = consensus_sn.fit_transform(X)

sklearn Compatibility
---------------------

NearestNeighbors (sklearn wrapper)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: treeclust.neighbors.NearestNeighbors

   Direct wrapper around sklearn's NearestNeighbors for compatibility.

kneighbors_graph (sklearn wrapper)  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: treeclust.neighbors.kneighbors_graph

   Direct wrapper around sklearn's kneighbors_graph function.

Usage Guidelines
----------------

**Choosing the Right Method**

- **KNeighbors**: Standard k-NN for most applications
- **MutualNearestNeighbors**: High-dimensional data or when standard k-NN struggles  
- **ConsensusNearestNeighbors**: When robustness is critical or parameters are uncertain
- **ConsensusMutualNearestNeighbors**: High-dimensional data requiring maximum robustness

**Parameter Selection**

**n_neighbors**:
   - Small datasets (< 500): 5-15 neighbors
   - Medium datasets (500-5000): 10-30 neighbors  
   - Large datasets (> 5000): 15-50 neighbors

**Distance Metrics**:
   - **euclidean**: Most common, works well for normalized data
   - **cosine**: Good for high-dimensional sparse data  
   - **manhattan**: Robust to outliers
   - **hamming**: For binary/categorical data

**Consensus Parameters**:
   - **n_bootstrap**: 20-100 depending on computational budget
   - **consensus_threshold**: 0.3-0.7, higher values = more conservative

**GPU Acceleration**

Some neighbors methods support GPU acceleration when available:

.. code-block:: python

   # Enable GPU if available
   knn = KNeighbors(n_neighbors=15, use_gpu=True)
   adjacency = knn.fit_transform(X)

**Integration with Clustering**

The neighbors module is designed to work seamlessly with the clustering module:

.. code-block:: python

   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import Leiden
   
   # Create adjacency matrix
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   # Perform clustering
   leiden = Leiden(resolution=1.0)
   labels = leiden.fit_predict(adjacency)