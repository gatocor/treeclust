Decomposition (:mod:`treeclust.decomposition`)
=============================================

.. automodule:: treeclust.decomposition

The decomposition module provides dimensionality reduction and manifold learning algorithms
with sklearn compatibility and optional GPU acceleration.

Dimensionality Reduction Methods
--------------------------------

The decomposition module includes various algorithms for reducing data dimensionality
before clustering or for visualization purposes:

**UMAP** (:mod:`treeclust.decomposition.umap`)
   Uniform Manifold Approximation and Projection for dimensionality reduction.

**t-SNE** (:mod:`treeclust.decomposition.tsne`) 
   t-Distributed Stochastic Neighbor Embedding for visualization.

**sklearn Integration** (:mod:`treeclust.decomposition.sklearn`)
   Wrappers and extensions for scikit-learn decomposition methods.

**VAE** (:mod:`treeclust.decomposition.vae`)
   Variational Autoencoder implementations for nonlinear dimensionality reduction.

**scVI** (:mod:`treeclust.decomposition.scvi`)
   Single-cell Variational Inference methods for biological data.

Usage Guidelines
----------------

**Choosing the Right Method**

- **UMAP**: Best for preserving both local and global structure, fast
- **t-SNE**: Good for visualization, preserves local structure well  
- **sklearn methods (PCA, etc.)**: Linear methods, interpretable, very fast
- **VAE**: Nonlinear, generative, good for complex data distributions
- **scVI**: Specialized for single-cell genomics data

**Integration with Clustering**

Dimensionality reduction is often used before clustering:

.. code-block:: python

   from treeclust.decomposition import UMAP  # (hypothetical)
   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import Leiden
   import numpy as np
   
   # High-dimensional data
   X = np.random.randn(1000, 100)
   
   # Reduce dimensionality
   umap = UMAP(n_components=15)
   X_reduced = umap.fit_transform(X)
   
   # Create adjacency in reduced space
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X_reduced)
   
   # Cluster
   leiden = Leiden(resolution=1.0)
   labels = leiden.fit_predict(adjacency)

**Best Practices**

1. **Choose appropriate dimensions**: 10-50 components for clustering, 2-3 for visualization
2. **Normalize data**: Most methods benefit from standardized input
3. **Consider computational cost**: Linear methods scale better than nonlinear
4. **Validate results**: Check that important structure is preserved after reduction

Note
----

Detailed API documentation for each decomposition method will be added as the module
is further developed. The module currently supports basic sklearn integration and is
being extended with additional specialized methods.