Metrics (:mod:`treeclust.metrics`)
==================================

.. automodule:: treeclust.metrics

The metrics module provides clustering quality metrics and evaluation tools
for assessing clustering performance and stability.

Clustering Quality Metrics
---------------------------

connectivity_probability
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: treeclust.metrics.connectivity_probability

   Compute the probability of connectivity between data points in clustering results.
   
   **Key Features:**
   
   - Measures clustering stability and quality
   - Useful for comparing different clustering solutions
   - Provides insights into cluster connectivity patterns
   - Can be used for parameter selection

   **Example:**
   
   .. code-block:: python
   
      from treeclust.metrics import connectivity_probability
      from treeclust.clustering import Leiden
      import numpy as np
      
      # Get clustering results
      labels = leiden.fit_predict(adjacency)
      
      # Compute connectivity probability
      conn_prob = connectivity_probability(
          labels=labels,
          adjacency_matrix=adjacency
      )
      
      print(f"Connectivity probability: {conn_prob:.3f}")

Usage Guidelines
----------------

**Metric Interpretation**

**Connectivity Probability**:
   - Values close to 1.0: High intra-cluster connectivity, good clustering
   - Values close to 0.5: Random connectivity, poor clustering  
   - Values close to 0.0: Low connectivity, potential over-clustering

**Integration with Other Modules**

The metrics module works seamlessly with clustering results:

.. code-block:: python

   from treeclust.clustering import MultiresolutionLeiden
   from treeclust.metrics import connectivity_probability
   from treeclust.neighbors import KNeighbors
   import numpy as np
   
   # Create data and adjacency
   X = np.random.randn(500, 30)
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   # Multi-resolution clustering
   mr_leiden = MultiresolutionLeiden(
       resolution_values=[0.5, 1.0, 1.5, 2.0]
   )
   results = mr_leiden.fit_predict(adjacency)
   
   # Evaluate each resolution
   for resolution, labels in results.items():
       if labels is not None:
           conn_prob = connectivity_probability(labels, adjacency)
           print(f"Resolution {resolution}: connectivity = {conn_prob:.3f}")

**Best Practices**

1. **Compare multiple metrics**: Use connectivity probability alongside other metrics like modularity
2. **Consider data characteristics**: Some metrics work better for different data types
3. **Validate across resolutions**: Check metric trends across parameter ranges
4. **Combine with consistency**: Use metrics together with consistency checking for robust evaluation