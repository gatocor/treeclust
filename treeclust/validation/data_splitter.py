import scipy.sparse as sp
import scipy.stats as stats

class PoissonDESplitter:
    """
    PoissonDESplitter class for binomial bootstrapping of count matrices.
    
    This class handles bootstrapping of integer count matrices by sampling each
    entry with a binomial distribution. For each entry in the matrix, the bootstrap
    sample contains a binomial(n=original_count, p=epsilon) value, and the remaining
    counts are assigned to the test set.
    
    This is particularly useful for differential expression analysis where you want
    to create bootstrap samples while preserving the count structure of the data.
    
    Features:
    - Binomial sampling of count matrices
    - Support for both dense and sparse matrices  
    - Configurable sampling probability (epsilon)
    - Automatic validation of input matrices
    """
    
    def __init__(self, epsilon: float = 0.8):
        """
        Initialize the PoissonDESplitter class.
        
        Parameters:
        -----------
        epsilon : float, default=0.8
            Probability parameter for binomial sampling (0 < epsilon <= 1).
            Higher values retain more of the original counts in the bootstrap sample.
        """
        # Validate epsilon parameter  
        if not (0 < epsilon <= 1.0):
            raise ValueError(f"epsilon must be between 0 and 1, got {epsilon}")
        
        self.epsilon = epsilon

    def sample(
        self, 
        X: Union[np.ndarray, sp.spmatrix]
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], Union[np.ndarray, sp.spmatrix]]:
        """
        Create a binomial bootstrap sample from a count matrix.
        
        For each entry in the input matrix, samples a binomial distribution 
        with n=original_count and p=epsilon. The bootstrap sample contains
        the sampled counts, and the test set contains the remaining counts.
        
        Parameters:
        -----------
        X : Union[np.ndarray, sp.spmatrix]
            Input count matrix. Must contain non-negative integer values.
            Can be either dense (numpy array) or sparse (scipy sparse matrix).
            
        Returns:
        --------
        Tuple[Union[np.ndarray, sp.spmatrix], Union[np.ndarray, sp.spmatrix]]
            - X_bootstrap: Bootstrap sample with binomial(n=X[i,j], p=epsilon) counts
            - X_test: Test set with remaining counts (X - X_bootstrap)
            
        Raises:
        -------
        ValueError
            If X contains negative values or non-integer values
        TypeError
            If X is not a numpy array or scipy sparse matrix
        """
        # Validate input type
        if not (isinstance(X, np.ndarray) or sp.issparse(X)):
            raise TypeError(f"X must be a numpy array or scipy sparse matrix, got {type(X)}")
        
        # Handle sparse matrices
        if sp.issparse(X):
            return self._sample_sparse(X)
        else:
            return self._sample_dense(X)
    
    def _validate_matrix(self, X: np.ndarray) -> None:
        """Validate that matrix contains non-negative integer values."""
        # Check for negative values
        if np.any(X < 0):
            raise ValueError("Input matrix must contain non-negative values")
        
        # Check for integer values (allowing for floating point representation of integers)
        if not np.allclose(X, X.astype(int)):
            raise ValueError("Input matrix must contain integer values")
    
    def _sample_dense(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from a dense matrix."""
        self._validate_matrix(X)
        
        # Convert to integer type for binomial sampling
        X_int = X.astype(int)
        
        # Handle edge case where X contains zeros
        mask = X_int > 0
        X_bootstrap = np.zeros_like(X_int)
        
        # Only sample where we have positive counts
        if np.any(mask):
            X_bootstrap[mask] = stats.binom.rvs(X_int[mask], self.epsilon)
        
        X_test = X_int - X_bootstrap
        
        return X_bootstrap, X_test
    
    def _sample_sparse(self, X: sp.spmatrix) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Sample from a sparse matrix."""
        # Convert to CSR format for efficient processing
        X_csr = X.tocsr()
        
        # Validate non-zero entries
        if np.any(X_csr.data < 0):
            raise ValueError("Input matrix must contain non-negative values")
        
        if not np.allclose(X_csr.data, X_csr.data.astype(int)):
            raise ValueError("Input matrix must contain integer values")
        
        # Convert data to integer
        X_csr.data = X_csr.data.astype(int)
        
        # Sample only non-zero entries
        bootstrap_data = stats.binom.rvs(X_csr.data, self.epsilon)
        test_data = X_csr.data - bootstrap_data
        
        # Create bootstrap matrix
        X_bootstrap = sp.csr_matrix(
            (bootstrap_data, X_csr.indices, X_csr.indptr), 
            shape=X_csr.shape
        )
        
        # Create test matrix  
        X_test = sp.csr_matrix(
            (test_data, X_csr.indices, X_csr.indptr),
            shape=X_csr.shape
        )
        
        # Convert back to original sparse format if needed
        if not isinstance(X, sp.csr_matrix):
            X_bootstrap = X_bootstrap.asformat(X.format)
            X_test = X_test.asformat(X.format)
        
        return X_bootstrap, X_test

    def __repr__(self) -> str:
        """String representation of the PoissonDESplitter object."""
        return f"PoissonDESplitter(epsilon={self.epsilon})"

class NegativeBinomialDESplitter:
    """
    NegativeBinomialDESplitter class for Dirichlet-Multinomial bootstrapping of count matrices.
    
    This class handles bootstrapping of integer count matrices by sampling each
    entry with a Dirichlet-Multinomial distribution. For each entry in the matrix, 
    the bootstrap sample is drawn from a multinomial distribution with probabilities
    determined by a Dirichlet distribution with concentration parameters [epsilon, 1-epsilon].
    
    This approach is particularly useful for negative binomial-distributed count data
    and provides a more flexible alternative to simple binomial sampling by allowing
    for overdispersion in the sampling process.
    
    Features:
    - Dirichlet-Multinomial sampling of count matrices
    - Support for both dense and sparse matrices  
    - Configurable sampling weight (epsilon)
    - Automatic validation of input matrices
    """
    
    def __init__(self, epsilon: float = 0.8):
        """
        Initialize the NegativeBinomialDESplitter class.
        
        Parameters:
        -----------
        epsilon : float, default=0.8
            Weight parameter for Dirichlet-Multinomial sampling (0 < epsilon <= 1).
            Higher values favor the bootstrap sample over the test set.
            The Dirichlet concentration parameters are [epsilon, 1-epsilon].
        """
        # Validate epsilon parameter  
        if not (0 < epsilon <= 1.0):
            raise ValueError(f"epsilon must be between 0 and 1, got {epsilon}")
        
        self.epsilon = epsilon

    def sample(
        self, 
        X: Union[np.ndarray, sp.spmatrix]
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], Union[np.ndarray, sp.spmatrix]]:
        """
        Create a Dirichlet-Multinomial bootstrap sample from a count matrix.
        
        For each entry in the input matrix, samples a multinomial distribution 
        with probabilities drawn from a Dirichlet distribution with concentration
        parameters [epsilon, 1-epsilon]. The bootstrap sample contains the first
        component and the test set contains the second component.
        
        Parameters:
        -----------
        X : Union[np.ndarray, sp.spmatrix]
            Input count matrix. Must contain non-negative integer values.
            Can be either dense (numpy array) or sparse (scipy sparse matrix).
            
        Returns:
        --------
        Tuple[Union[np.ndarray, sp.spmatrix], Union[np.ndarray, sp.spmatrix]]
            - X_bootstrap: Bootstrap sample from Dirichlet-Multinomial
            - X_test: Test set with remaining counts (X - X_bootstrap)
            
        Raises:
        -------
        ValueError
            If X contains negative values or non-integer values
        TypeError
            If X is not a numpy array or scipy sparse matrix
        """
        # Validate input type
        if not (isinstance(X, np.ndarray) or sp.issparse(X)):
            raise TypeError(f"X must be a numpy array or scipy sparse matrix, got {type(X)}")
        
        # Handle sparse matrices
        if sp.issparse(X):
            return self._sample_sparse(X)
        else:
            return self._sample_dense(X)
    
    def _validate_matrix(self, X: np.ndarray) -> None:
        """Validate that matrix contains non-negative integer values."""
        # Check for negative values
        if np.any(X < 0):
            raise ValueError("Input matrix must contain non-negative values")
        
        # Check for integer values (allowing for floating point representation of integers)
        if not np.allclose(X, X.astype(int)):
            raise ValueError("Input matrix must contain integer values")
    
    def _sample_dense(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from a dense matrix using Dirichlet-Multinomial."""
        self._validate_matrix(X)
        
        # Convert to integer type for multinomial sampling
        X_int = X.astype(int)
        
        # Initialize bootstrap matrix
        X_bootstrap = np.zeros_like(X_int)
        
        # Handle each non-zero entry
        mask = X_int > 0
        if np.any(mask):
            # Get positions of non-zero entries
            nonzero_idx = np.where(mask)
            
            for i, j in zip(nonzero_idx[0], nonzero_idx[1]):
                n_total = X_int[i, j]
                
                # Draw probability from Dirichlet(epsilon, 1-epsilon)
                # This gives us the probability of assigning to bootstrap vs test
                alpha = np.array([self.epsilon, 1 - self.epsilon])
                p = stats.dirichlet.rvs(alpha)[0]  # Get the first (and only) sample
                
                # Sample from multinomial with these probabilities
                # multinomial returns array of counts for each category
                counts = stats.multinomial.rvs(n_total, p)
                X_bootstrap[i, j] = counts[0]  # First category goes to bootstrap
        
        X_test = X_int - X_bootstrap
        
        return X_bootstrap, X_test
    
    def _sample_sparse(self, X: sp.spmatrix) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Sample from a sparse matrix using Dirichlet-Multinomial."""
        # Convert to CSR format for efficient processing
        X_csr = X.tocsr()
        
        # Validate non-zero entries
        if np.any(X_csr.data < 0):
            raise ValueError("Input matrix must contain non-negative values")
        
        if not np.allclose(X_csr.data, X_csr.data.astype(int)):
            raise ValueError("Input matrix must contain integer values")
        
        # Convert data to integer
        X_csr.data = X_csr.data.astype(int)
        
        # Sample only non-zero entries using Dirichlet-Multinomial
        bootstrap_data = np.zeros_like(X_csr.data)
        
        for idx, n_total in enumerate(X_csr.data):
            if n_total > 0:
                # Draw probability from Dirichlet(epsilon, 1-epsilon)
                alpha = np.array([self.epsilon, 1 - self.epsilon])
                p = stats.dirichlet.rvs(alpha)[0]  # Get the first (and only) sample
                
                # Sample from multinomial with these probabilities
                counts = stats.multinomial.rvs(n_total, p)
                bootstrap_data[idx] = counts[0]  # First category goes to bootstrap
        
        test_data = X_csr.data - bootstrap_data
        
        # Create bootstrap matrix
        X_bootstrap = sp.csr_matrix(
            (bootstrap_data, X_csr.indices, X_csr.indptr), 
            shape=X_csr.shape
        )
        
        # Create test matrix  
        X_test = sp.csr_matrix(
            (test_data, X_csr.indices, X_csr.indptr),
            shape=X_csr.shape
        )
        
        # Convert back to original sparse format if needed
        if not isinstance(X, sp.csr_matrix):
            X_bootstrap = X_bootstrap.asformat(X.format)
            X_test = X_test.asformat(X.format)
        
        return X_bootstrap, X_test

    def __repr__(self) -> str:
        """String representation of the NegativeBinomialDESplitter object."""
        return f"NegativeBinomialDESplitter(epsilon={self.epsilon})"