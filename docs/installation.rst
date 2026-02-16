Installation Guide
==================

Requirements
------------

TreeClust requires Python 3.8+ and includes the following dependency categories:

Core Dependencies (Always Installed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- numpy >= 1.19.0
- scipy >= 1.5.0 
- matplotlib >= 3.3.0
- pandas >= 1.1.0
- seaborn >= 0.11.0
- tqdm >= 4.50.0
- scikit-learn >= 0.24.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

**Clustering Algorithms** (``clustering``):
- leidenalg >= 0.8.0 - Leiden clustering algorithm
- igraph >= 0.9.0 - Network analysis and graph algorithms
- scikit-network >= 0.24.0 - Additional network algorithms

**GPU Acceleration** (``gpu``):
- cupy >= 8.0.0 - GPU-accelerated computing (NVIDIA only)
- rapids-cudf >= 21.06.0 - GPU DataFrames (NVIDIA only)
- cugraph >= 21.06.0 - GPU graph analytics (NVIDIA only)

**Biological Data Analysis** (``bio``):
- anndata >= 0.7.0 - Annotated data structures for single-cell analysis
- scanpy >= 1.7.0 - Single-cell analysis in Python
- scvi-tools >= 0.14.0 - Deep generative models for single-cell data

**Enhanced Plotting** (``plotting``):
- plotly >= 4.14.0 - Interactive plotting
- bokeh >= 2.3.0 - Interactive visualization library
- networkx >= 2.5.0 - Network analysis and visualization
- cairocffi >= 1.2.0 - 2D graphics library
- pygraphviz >= 1.7.0 - GraphViz interface for network layouts

Installation Methods
--------------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install treeclust with core dependencies only:

.. code-block:: bash

   pip install treeclust

Install with Specific Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install with clustering algorithms (recommended for most users)
   pip install "treeclust[clustering]"

   # Install with biological data analysis tools
   pip install "treeclust[bio]"

   # Install with enhanced plotting capabilities
   pip install "treeclust[plotting]"

   # Install with GPU acceleration (NVIDIA GPUs only)
   pip install "treeclust[gpu]"

   # Install with development tools
   pip install "treeclust[dev]"

   # Install with documentation building tools
   pip install "treeclust[docs]"

Install with Multiple Option Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clustering + biological analysis
   pip install "treeclust[clustering,bio]"

   # Full installation with all optional dependencies
   pip install "treeclust[all]"

   # Development setup
   pip install "treeclust[clustering,dev,docs]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/gatocor/treeclust.git
   cd treeclust
   
   # Basic development install
   pip install -e .
   
   # Development install with all dependencies
   pip install -e ".[all,dev]"

Conda Installation
~~~~~~~~~~~~~~~~~~

If you prefer conda, you can create an environment using the provided environment file:

.. code-block:: bash

   git clone https://github.com/gatocor/treeclust.git
   cd treeclust
   conda env create -f environment.yml
   conda activate treeclust
   
   # Install treeclust in development mode
   pip install -e ".[clustering]"

Platform-Specific Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Support (NVIDIA only)**:

.. code-block:: bash

   # Ensure CUDA is installed first
   pip install "treeclust[gpu]"

**Biological Analysis**:

.. code-block:: bash

   # For single-cell genomics workflows
   pip install "treeclust[bio,clustering]"

**Complete Scientific Setup**:

.. code-block:: bash

   # Full scientific computing stack
   pip install "treeclust[all]"

Installation Verification
~~~~~~~~~~~~~~~~~~~~~~~~~

To verify your installation works correctly:

.. code-block:: python

   import treeclust
   print(f"TreeClust version: {treeclust.__version__}")

   # Test basic functionality
   from treeclust.neighbors import KNeighbors
   import numpy as np

   # Generate test data
   X = np.random.randn(100, 10)
   
   # Create adjacency matrix (should work with core dependencies)
   knn = KNeighbors(n_neighbors=5)
   adjacency = knn.fit_transform(X)
   
   print("Basic installation verified!")

**Test Optional Dependencies**:

.. code-block:: python

   # Test clustering algorithms (optional)
   try:
       from treeclust.clustering import Leiden
       print("✓ Clustering algorithms available")
   except ImportError:
       print("✗ Clustering algorithms not available (install with [clustering])")

   # Test GPU support (optional)
   try:
       import cupy
       print("✓ GPU acceleration available")
   except ImportError:
       print("✗ GPU acceleration not available (install with [gpu])")

   # Test biological data tools (optional)  
   try:
       import anndata
       print("✓ Biological data tools available")
   except ImportError:
       print("✗ Biological data tools not available (install with [bio])")

Recommended Installation Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For most users**:
   ``pip install "treeclust[clustering]"``

**For single-cell analysis**:
   ``pip install "treeclust[clustering,bio]"``

**For interactive analysis**:
   ``pip install "treeclust[clustering,plotting]"``

**For high-performance computing**:
   ``pip install "treeclust[clustering,gpu]"``

**For developers**:
   ``pip install -e ".[all,dev,docs]"``

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'leidenalg'**
   Install clustering dependencies: ``pip install "treeclust[clustering]"``

**ImportError: No module named 'igraph'**
   Install clustering dependencies: ``pip install "treeclust[clustering]"``

**ImportError: No module named 'anndata'**
   Install biological analysis tools: ``pip install "treeclust[bio]"``

**CUDA/GPU issues**
   Make sure you have compatible CUDA drivers and install GPU dependencies: ``pip install "treeclust[gpu]"``

**Build errors on Windows**
   Some packages may require Visual Studio Build Tools. Install from Microsoft's website.
   For igraph specifically, try: ``conda install -c conda-forge igraph``

**Build errors on macOS**
   You may need Xcode command line tools: ``xcode-select --install``
   For Cairo/GraphViz issues: ``brew install cairo graphviz``

**Memory issues with large datasets**
   Consider installing GPU acceleration: ``pip install "treeclust[gpu]"``

**Missing scientific packages**
   Install the full scientific stack: ``pip install "treeclust[all]"``

Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows**:
   - Use conda for packages with C dependencies: ``conda install -c conda-forge igraph leidenalg``
   - For GPU support, ensure CUDA toolkit is properly installed

**macOS**:
   - Install Homebrew dependencies: ``brew install cairo graphviz``
   - For M1/M2 Macs, some packages may need conda-forge: ``conda install -c conda-forge igraph``

**Linux**:
   - Install system dependencies: ``sudo apt-get install libcairo2-dev libgraphviz-dev``
   - For GPU support, ensure CUDA drivers and toolkit are installed

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~

If you encounter dependency conflicts:

.. code-block:: bash

   # Create a clean environment
   conda create -n treeclust python=3.9
   conda activate treeclust
   
   # Install with conda-forge for better compatibility
   conda install -c conda-forge numpy scipy matplotlib pandas
   pip install "treeclust[clustering]"

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**For large datasets**:

.. code-block:: bash

   # Install with GPU support
   pip install "treeclust[gpu]"

**For biological data**:

.. code-block:: bash

   # Install optimized scientific stack
   conda install -c conda-forge numpy scipy pandas
   pip install "treeclust[bio]"

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/gatocor/treeclust/issues>`_
2. Read the documentation thoroughly
3. Create a new issue with detailed error information