Validation (:mod:`treeclust.validation`)
=========================================

.. automodule:: treeclust.validation

The validation module provides tools for validating clustering results and assessing
clustering stability through data splitting and cross-validation approaches.

Data Splitting and Validation
------------------------------

data_splitter
~~~~~~~~~~~~~

.. automodule:: treeclust.validation.data_splitter

   Tools for splitting datasets to validate clustering stability and generalization.
   
   **Key Features:**
   
   - Split data for clustering validation
   - Assess clustering stability across data subsets  
   - Cross-validation approaches for clustering
   - Bootstrap sampling for robustness testing

   **Example Usage:**
   
   .. code-block:: python
   
      from treeclust.validation.data_splitter import split_data  # (hypothetical)
      from treeclust.clustering import Leiden
      import numpy as np
      
      # Original data
      X = np.random.randn(1000, 50)
      
      # Split for validation
      X_train, X_test, indices_train, indices_test = split_data(
          X, test_size=0.3, random_state=42
      )
      
      # Cluster on training data
      leiden = Leiden(resolution=1.0)
      train_labels = leiden.fit_predict(train_adjacency)
      
      # Validate on test data  
      test_labels = leiden.predict(test_adjacency)

Validation Strategies
--------------------

**Stability Validation**

The validation module supports several approaches to assess clustering stability:

1. **Train/Test Split**: Cluster on subset, validate on holdout
2. **Cross-Validation**: Multiple train/test splits for robust assessment  
3. **Bootstrap Validation**: Sample with replacement for stability testing
4. **Subsampling**: Random subsets to test parameter sensitivity

**Integration with Clustering**

Validation works seamlessly with the clustering module:

.. code-block:: python

   from treeclust.validation import cross_validate_clustering  # (hypothetical)
   from treeclust.clustering import MultiresolutionLeiden
   from treeclust.neighbors import KNeighbors
   import numpy as np
   
   # Data and adjacency
   X = np.random.randn(500, 30)
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   # Cross-validate multiresolution clustering
   cv_results = cross_validate_clustering(
       MultiresolutionLeiden(resolution_values=[0.5, 1.0, 2.0]),
       adjacency,
       cv=5,
       metrics=['ari', 'nmi']
   )

Usage Guidelines
----------------

**When to Use Validation**

- **Parameter Selection**: Compare clustering parameters objectively
- **Method Comparison**: Evaluate different clustering algorithms
- **Stability Assessment**: Ensure clustering is not overfitted to data
- **Publication**: Provide rigorous evaluation for research

**Best Practices**

1. **Use appropriate splits**: Ensure train/test splits preserve data structure
2. **Multiple validation runs**: Use cross-validation or bootstrap for robustness
3. **Consistent preprocessing**: Apply same preprocessing to all data splits
4. **Report confidence intervals**: Show uncertainty in validation metrics

Note
----

The validation module is under active development. Additional validation tools and 
cross-validation strategies will be added in future releases.