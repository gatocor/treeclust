Clustering (:mod:`treeclust.clustering`)
==========================================

.. automodule:: treeclust.clustering

The clustering module provides various clustering algorithms with sklearn-style interfaces, 
consistency checking for stochastic methods, and specialized parameter tuning capabilities.

Core Clustering Classes
-----------------------

Leiden Clustering
~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.clustering.Leiden
   :members:
   :inherited-members:
   :show-inheritance:

   The Leiden clustering algorithm implementation with consistency checking support.
   
   **Key Features:**
   
   - Requires adjacency matrix input (no internal connectivity computation)
   - Consistency checking with multiple repetitions for stochastic validation
   - Support for different quality functions and resolution parameters
   - Integration with igraph for efficient clustering

   **Example:**
   
   .. code-block:: python
   
      from treeclust.clustering import Leiden
      from treeclust.neighbors import KNeighbors
      import numpy as np
      
      # Create adjacency matrix
      X = np.random.randn(500, 20)
      knn = KNeighbors(n_neighbors=10, mode='connectivity')
      adjacency = knn.fit_transform(X)
      
      # Perform Leiden clustering with consistency checking
      leiden = Leiden(
          resolution=1.0,
          n_repetitions=5,
          consistency_metric='ari'
      )
      labels = leiden.fit_predict(adjacency)
      
      # Check consistency
      consistency = leiden.get_consistency_summary()
      print(f"Mean consistency: {consistency['mean_consistency']:.3f}")

Louvain Clustering
~~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.clustering.Louvain
   :members:
   :inherited-members:
   :show-inheritance:

   The Louvain clustering algorithm implementation with consistency checking support.
   
   **Key Features:**
   
   - Requires adjacency matrix input (no internal connectivity computation) 
   - Consistency checking with multiple repetitions for stochastic validation
   - Support for different quality functions and resolution parameters
   - Integration with igraph for efficient clustering

   **Example:**
   
   .. code-block:: python
   
      from treeclust.clustering import Louvain
      from treeclust.neighbors import KNeighbors
      import numpy as np
      
      # Create adjacency matrix
      X = np.random.randn(500, 20)
      knn = KNeighbors(n_neighbors=10, mode='connectivity')
      adjacency = knn.fit_transform(X)
      
      # Perform Louvain clustering with consistency checking
      louvain = Louvain(
          resolution=1.0, 
          n_repetitions=5,
          consistency_metric='ari'
      )
      labels = louvain.fit_predict(adjacency)
      
      # Check consistency
      consistency = louvain.get_consistency_summary()
      print(f"Mean consistency: {consistency['mean_consistency']:.3f}")

Parameter Tuning Classes
------------------------

Multiscale Clustering
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.clustering.Multiscale
   :members:
   :inherited-members:
   :show-inheritance:

   Generic parameter tuning class that can optimize any parameter of any clustering algorithm.
   
   **Key Features:**
   
   - Accepts constructed clustering instances (not classes)
   - Can tune any parameter across multiple values
   - Stores all clustering results for comparison
   - sklearn-style interface for easy integration

   **Example:**
   
   .. code-block:: python
   
      from treeclust.clustering import Multiscale, Leiden
      import numpy as np
      
      # Create base clustering instance
      leiden_instance = Leiden(resolution=1.0, random_state=42)
      
      # Create multiscale tuning for resolution parameter
      multiscale = Multiscale(
          clustering_instance=leiden_instance,
          parameter_name='resolution',
          parameter_values=[0.1, 0.5, 1.0, 2.0, 5.0]
      )
      
      # Fit and get results
      results = multiscale.fit_predict(adjacency)
      best_clusterer = multiscale.get_best_clusterer('silhouette')

MultiresolutionLeiden
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.clustering.MultiresolutionLeiden
   :members:
   :inherited-members:
   :show-inheritance:

   Specialized resolution parameter tuning for Leiden clustering with comprehensive analysis.
   
   **Key Features:**
   
   - Resolution-specific parameter tuning (specialized for Leiden)
   - Integration with consistency checking features
   - Comprehensive metrics and analysis methods
   - Methods to find optimal resolution based on different criteria
   - Modularity computation and range analysis

   **Example:**
   
   .. code-block:: python
   
      from treeclust.clustering import MultiresolutionLeiden
      import numpy as np
      
      # Test Leiden across different resolutions
      mr_leiden = MultiresolutionLeiden(
          resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0],
          n_repetitions=5,
          consistency_metric='ari'
      )
      
      # Fit and analyze
      results = mr_leiden.fit_predict(adjacency)
      
      # Find optimal resolution
      best_resolution = mr_leiden.get_best_resolution('consistency')
      best_labels = mr_leiden.get_labels(best_resolution)
      
      # Get comprehensive summary
      summary = mr_leiden.get_resolution_range_summary()

MultiresolutionLouvain
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: treeclust.clustering.MultiresolutionLouvain
   :members:
   :inherited-members:
   :show-inheritance:

   Specialized resolution parameter tuning for Louvain clustering with comprehensive analysis.
   
   **Key Features:**
   
   - Resolution-specific parameter tuning (specialized for Louvain)
   - Integration with consistency checking features  
   - Comprehensive metrics and analysis methods
   - Methods to find optimal resolution based on different criteria
   - Modularity computation and range analysis

   **Example:**
   
   .. code-block:: python
   
      from treeclust.clustering import MultiresolutionLouvain
      import numpy as np
      
      # Test Louvain across different resolutions
      mr_louvain = MultiresolutionLouvain(
          resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0],
          n_repetitions=5,
          consistency_metric='ari'
      )
      
      # Fit and analyze
      results = mr_louvain.fit_predict(adjacency)
      
      # Find optimal resolution  
      best_resolution = mr_louvain.get_best_resolution('consistency')
      best_labels = mr_louvain.get_labels(best_resolution)
      
      # Get comprehensive summary
      summary = mr_louvain.get_resolution_range_summary()

Consistency Metrics
-------------------

Both Leiden and Louvain clustering support consistency checking when ``n_repetitions > 1``.
The following metrics are available for measuring consistency between multiple runs:

- **ari** (Adjusted Rand Index): Measures clustering similarity accounting for chance
- **ami** (Adjusted Mutual Information): Information-theoretic clustering similarity  
- **nmi** (Normalized Mutual Information): Normalized information-theoretic similarity
- **homogeneity**: Measures if clusters contain only data points from single class
- **completeness**: Measures if data points from same class are in same cluster  
- **v_measure**: Harmonic mean of homogeneity and completeness

Usage Tips
----------

**Adjacency Matrix Requirements**

All clustering classes in this module require pre-computed adjacency matrices as input.
Use the ``treeclust.neighbors`` module to create adjacency matrices:

.. code-block:: python

   from treeclust.neighbors import KNeighbors
   
   # Create k-nearest neighbors adjacency
   knn = KNeighbors(n_neighbors=15, mode='connectivity') 
   adjacency = knn.fit_transform(X)

**Choosing Resolution Values**

For multiresolution analysis, choose resolution values that span the range of interest:

- **Low values (0.1-0.5)**: Favor fewer, larger clusters
- **Medium values (0.8-2.0)**: Balanced cluster sizes  
- **High values (2.0-10.0)**: Favor many, smaller clusters

**Consistency Checking**

Use consistency checking to validate results from stochastic algorithms:

- **n_repetitions=1**: No consistency checking (fastest)
- **n_repetitions=5-10**: Good balance of validation and speed
- **n_repetitions=20+**: Thorough validation for critical applications