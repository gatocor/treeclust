Bootstrapping (:mod:`treeclust.bootstrapping`)
==============================================

.. automodule:: treeclust.bootstrapping

The bootstrapping module provides tools for bootstrap sampling of data and parameters
to assess clustering stability and robustness.

Bootstrap Methods
-----------------

data_bootstrapper
~~~~~~~~~~~~~~~~~

.. automodule:: treeclust.bootstrapping.data_bootstrapper

   Bootstrap sampling of data points to assess clustering stability.
   
   **Key Features:**
   
   - Sample data points with replacement for robustness testing
   - Generate multiple bootstrap samples for ensemble analysis
   - Assess clustering stability across data variations
   - Integration with consensus methods

   **Example Usage:**
   
   .. code-block:: python
   
      from treeclust.bootstrapping.data_bootstrapper import DataBootstrapper  # (hypothetical)
      from treeclust.clustering import Leiden
      import numpy as np
      
      # Create bootstrap sampler
      bootstrapper = DataBootstrapper(
          n_bootstrap=50,
          sample_fraction=0.8,
          random_state=42
      )
      
      # Generate bootstrap samples and cluster each
      X = np.random.randn(500, 30)
      
      bootstrap_results = []
      for X_boot, indices_boot in bootstrapper.generate_samples(X):
          # Cluster bootstrap sample
          leiden = Leiden(resolution=1.0)
          labels_boot = leiden.fit_predict(adjacency_boot)
          bootstrap_results.append(labels_boot)

parameter_bootstrapper  
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: treeclust.bootstrapping.parameter_bootstrapper

   Bootstrap sampling of clustering parameters to explore parameter sensitivity.
   
   **Key Features:**
   
   - Sample clustering parameters from specified distributions
   - Assess parameter sensitivity and robustness
   - Ensemble clustering across parameter ranges
   - Uncertainty quantification for clustering results

   **Example Usage:**
   
   .. code-block:: python
   
      from treeclust.bootstrapping.parameter_bootstrapper import ParameterBootstrapper  # (hypothetical)
      from treeclust.clustering import Leiden
      import numpy as np
      
      # Create parameter bootstrapper
      param_bootstrapper = ParameterBootstrapper(
          parameter_ranges={
              'resolution': (0.5, 2.0),
              'n_iterations': (1, 5)
          },
          n_bootstrap=30,
          random_state=42
      )
      
      # Sample parameters and cluster
      for params in param_bootstrapper.generate_parameters():
          leiden = Leiden(**params)
          labels = leiden.fit_predict(adjacency)

Bootstrap Integration
---------------------

**Consensus Clustering**

Bootstrap methods integrate with consensus approaches:

.. code-block:: python

   from treeclust.bootstrapping import DataBootstrapper  # (hypothetical)
   from treeclust.neighbors import ConsensusNearestNeighbors
   from treeclust.clustering import Leiden
   import numpy as np
   
   # Bootstrap data and neighbors
   consensus_knn = ConsensusNearestNeighbors(
       n_neighbors_list=[10, 15, 20],
       n_bootstrap=50
   )
   
   # Create consensus adjacency
   consensus_adj = consensus_knn.fit_transform(X)
   
   # Cluster consensus result
   leiden = Leiden(resolution=1.0)
   labels = leiden.fit_predict(consensus_adj)

**Multiresolution Bootstrap**

Combine with multiresolution methods:

.. code-block:: python

   from treeclust.bootstrapping import ParameterBootstrapper  # (hypothetical)
   from treeclust.clustering import MultiresolutionLeiden
   import numpy as np
   
   # Bootstrap resolution parameters
   param_bootstrap = ParameterBootstrapper(
       parameter_ranges={'resolution': (0.1, 5.0)},
       n_bootstrap=20
   )
   
   # Create ensemble of multiresolution results
   ensemble_results = []
   for params in param_bootstrap.generate_parameters():
       mr_leiden = MultiresolutionLeiden(
           resolution_values=[params['resolution']]
       )
       result = mr_leiden.fit_predict(adjacency)
       ensemble_results.append(result)

Usage Guidelines
----------------

**Bootstrap Sample Size**

- **Small datasets (< 500)**: 20-50 bootstrap samples
- **Medium datasets (500-5000)**: 30-100 bootstrap samples  
- **Large datasets (> 5000)**: 50-200 bootstrap samples

**Parameter Bootstrap Ranges**

- **Resolution**: Typically 0.1-10.0, focus on range of interest
- **Neighbors**: ±50% of original value
- **Other parameters**: Based on domain knowledge and prior exploration

**Statistical Interpretation**

- **Stability**: High agreement across bootstrap samples indicates stable clustering
- **Uncertainty**: Variability across samples quantifies clustering uncertainty
- **Robustness**: Consistent results across parameter ranges show robustness

**Best Practices**

1. **Set random seeds**: For reproducible bootstrap sampling
2. **Monitor convergence**: Ensure enough bootstrap samples for stable estimates
3. **Validate assumptions**: Check that bootstrap assumptions are reasonable
4. **Report confidence**: Provide uncertainty estimates with clustering results

Note
----

The bootstrapping module is under development. Additional bootstrap strategies and 
statistical analysis tools will be added in future releases.