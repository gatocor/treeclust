"""
Dimensionality reduction utilities for treeclust.

This module provides dimensionality reduction techniques including
Variational Autoencoders (VAE) with sklearn-style API.
"""

import numpy as np
import warnings
from typing import List, Optional, Union, Tuple
from abc import ABC, abstractmethod

# Import centralized availability flags
from .. import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    # If PyTorch is not available, create a dummy VAE class
    class VAE:
        """
        Dummy VAE class when PyTorch is not available.
        
        This class raises ImportError when instantiated to inform users
        that PyTorch is required for VAE functionality.
        """
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for VAE functionality. "
                "Please install PyTorch to use the VAE class."
            )
        
        def fit(self, *args, **kwargs):
            raise ImportError("PyTorch is required for VAE functionality.")
        
        def transform(self, *args, **kwargs):
            raise ImportError("PyTorch is required for VAE functionality.")
        
        def fit_transform(self, *args, **kwargs):
            raise ImportError("PyTorch is required for VAE functionality.")

else:
    # Only define the real VAE class when PyTorch is available
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class VAE(nn.Module):
        """
        Variational Autoencoder (VAE) with sklearn-style API.
        
        This class implements a VAE using PyTorch with a flexible architecture
        that allows customization of encoder/decoder layers and latent dimensions.
        The API follows sklearn conventions with fit, transform, and fit_transform methods.
    
    Features:
    - Customizable encoder/decoder architecture
    - Configurable latent dimension
    - Support for different activation functions
    - Batch training with progress monitoring
    - GPU support when available
    - sklearn-style API for easy integration
    """
    
    def __init__(
        self,
        latent_dim: int = 10,
        encoder_layers: Optional[List[int]] = None,
        decoder_layers: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        beta: float = 1.0,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the VAE model.
        
        Parameters
        ----------
        latent_dim : int, default=10
            Dimension of the latent space.
        encoder_layers : List[int], optional
            Hidden layer sizes for the encoder. If None, uses [512, 256].
        decoder_layers : List[int], optional
            Hidden layer sizes for the decoder. If None, uses [256, 512].
        activation : str, default='relu'
            Activation function to use ('relu', 'tanh', 'sigmoid', 'leaky_relu').
        dropout : float, default=0.1
            Dropout probability for regularization.
        learning_rate : float, default=1e-3
            Learning rate for the optimizer.
        batch_size : int, default=64
            Batch size for training.
        epochs : int, default=100
            Number of training epochs.
        beta : float, default=1.0
            Beta parameter for β-VAE (controls KL divergence regularization).
        device : str, optional
            Device to use ('cpu', 'cuda'). If None, auto-detects.
        random_state : int, optional
            Random seed for reproducibility.
        verbose : bool, default=True
            Whether to print training progress.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for VAE functionality. "
                "Please install PyTorch to use the VAE class."
            )
            
        super(VAE, self).__init__()
    ):
        """
        Initialize the VAE.
        
        Parameters:
        -----------
        latent_dim : int, default=10
            Dimension of the latent space.
            
        encoder_layers : List[int], optional
            List of hidden layer sizes for encoder. If None, uses [512, 256].
            Example: [512, 256] creates encoder: input -> 512 -> 256 -> latent_dim
            
        decoder_layers : List[int], optional
            List of hidden layer sizes for decoder. If None, mirrors encoder.
            Example: [256, 512] creates decoder: latent_dim -> 256 -> 512 -> output
            
        activation : str, default='relu'
            Activation function to use. Options: 'relu', 'tanh', 'sigmoid', 'leaky_relu'
            
        dropout : float, default=0.1
            Dropout probability for regularization.
            
        learning_rate : float, default=1e-3
            Learning rate for optimization.
            
        batch_size : int, default=64
            Batch size for training.
            
        epochs : int, default=100
            Number of training epochs.
            
        beta : float, default=1.0
            Beta parameter for β-VAE (controls KL divergence weight).
            
        device : str, optional
            Device to use ('cpu', 'cuda', 'mps'). If None, auto-selects.
            
        random_state : int, optional
            Random seed for reproducibility.
            
        verbose : bool, default=True
            Whether to print training progress.
        """
        super(VAE, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for VAE. Install with: pip install torch"
            )
        
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers or [512, 256]
        self.decoder_layers = decoder_layers or list(reversed(self.encoder_layers))
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.verbose = verbose
        
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Get activation function
        self.activation_fn = self._get_activation_function(activation)
        
        # Initialize components (will be built when input dimension is known)
        self.input_dim = None
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        
        # Training history
        self.history_ = {
            'loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        
        self.is_fitted_ = False
        
    def _get_activation_function(self, activation: str):
        """Get PyTorch activation function."""
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Available: {list(activations.keys())}")
        
        return activations[activation]
    
    def _build_networks(self, input_dim: int):
        """Build encoder and decoder networks."""
        self.input_dim = input_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.encoder_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Latent layers (mean and log variance)
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in self.decoder_layers:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Move to device
        self.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
            
        Returns:
        --------
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log variance of latent distribution.
        """
        h = x
        
        # Pass through encoder layers
        for i in range(0, len(self.encoder), 2):
            h = self.encoder[i](h)  # Linear layer
            h = self.activation_fn(h)
            if i + 1 < len(self.encoder):
                h = self.encoder[i + 1](h)  # Dropout layer
        
        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log variance of latent distribution.
            
        Returns:
        --------
        z : torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output space.
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent vector.
            
        Returns:
        --------
        recon : torch.Tensor
            Reconstructed output.
        """
        h = z
        
        # Pass through decoder layers
        for i in range(0, len(self.decoder) - 1, 2):
            h = self.decoder[i](h)  # Linear layer
            h = self.activation_fn(h)
            if i + 1 < len(self.decoder) - 1:
                h = self.decoder[i + 1](h)  # Dropout layer
        
        # Output layer (no activation)
        recon = self.decoder[-1](h)
        return recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
            
        Returns:
        --------
        recon : torch.Tensor
            Reconstructed output.
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log variance of latent distribution.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Parameters:
        -----------
        recon_x : torch.Tensor
            Reconstructed input.
        x : torch.Tensor
            Original input.
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log variance of latent distribution.
            
        Returns:
        --------
        total_loss : torch.Tensor
            Total VAE loss.
        recon_loss : torch.Tensor
            Reconstruction loss.
        kl_loss : torch.Tensor
            KL divergence loss.
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'VAE':
        """
        Fit the VAE to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Ignored. Present for sklearn compatibility.
            
        Returns:
        --------
        self : VAE
            Returns self for method chaining.
        """
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X.float()
        
        # Build networks if not already built
        if self.encoder is None:
            self._build_networks(X_tensor.shape[1])
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = self.forward(data)
                
                # Compute loss
                loss, recon_loss, kl_loss = self.loss_function(recon_batch, data, mu, logvar)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            # Average losses over epoch
            avg_loss = epoch_loss / len(dataloader.dataset)
            avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
            avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
            
            # Store history
            self.history_['loss'].append(avg_loss)
            self.history_['recon_loss'].append(avg_recon_loss)
            self.history_['kl_loss'].append(avg_kl_loss)
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], '
                      f'Loss: {avg_loss:.4f}, '
                      f'Recon: {avg_recon_loss:.4f}, '
                      f'KL: {avg_kl_loss:.4f}')
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to latent space.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to transform of shape (n_samples, n_features).
            
        Returns:
        --------
        X_latent : np.ndarray
            Transformed data in latent space of shape (n_samples, latent_dim).
        """
        if not self.is_fitted_:
            raise ValueError("VAE must be fitted before transform. Call fit() first.")
        
        self.eval()
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X.float()
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            mu, logvar = self.encode(X_tensor)
            # Use mean for deterministic transformation
            z = mu
        
        return z.cpu().numpy()
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the VAE and transform the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Ignored. Present for sklearn compatibility.
            
        Returns:
        --------
        X_latent : np.ndarray
            Transformed data in latent space.
        """
        return self.fit(X, y).transform(X)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data through the VAE.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to reconstruct.
            
        Returns:
        --------
        X_recon : np.ndarray
            Reconstructed data.
        """
        if not self.is_fitted_:
            raise ValueError("VAE must be fitted before reconstruction. Call fit() first.")
        
        self.eval()
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X.float()
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            recon, _, _ = self.forward(X_tensor)
        
        return recon.cpu().numpy()
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate samples from the latent space.
        
        Parameters:
        -----------
        n_samples : int, default=1
            Number of samples to generate.
            
        Returns:
        --------
        samples : np.ndarray
            Generated samples.
        """
        if not self.is_fitted_:
            raise ValueError("VAE must be fitted before sampling. Call fit() first.")
        
        self.eval()
        
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            # Decode to data space
            samples = self.decode(z)
        
        return samples.cpu().numpy()
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn compatibility)."""
        return {
            'latent_dim': self.latent_dim,
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'beta': self.beta,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'VAE':
        """Set parameters for this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def __repr__(self) -> str:
        """String representation of the VAE."""
        return (f"VAE(latent_dim={self.latent_dim}, "
                f"encoder_layers={self.encoder_layers}, "
                f"decoder_layers={self.decoder_layers}, "
                f"activation='{self.activation}', "
                f"epochs={self.epochs}, "
                f"beta={self.beta}, "
                f"fitted={self.is_fitted_})")