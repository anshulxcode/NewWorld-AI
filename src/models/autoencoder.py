"""
Autoencoder for Unsupervised Anomaly and Novelty Detection in NetWorld-AI.
Trained exclusively on BENIGN traffic samples. High reconstruction error signals zero-day attacks or distribution shift.
Includes Monte Carlo (MC) Dropout for uncertainty estimation and confidence interval calculation.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    Symmetric Autoencoder architecture for network traffic novelty detection.
    Compresses input feature vector to a 16-dim latent bottleneck and reconstructs original metrics.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, dropout_prob: float = 0.1) -> None:
        """
        Initializes Autoencoder model.

        Args:
            input_dim: Number of input features.
            latent_dim: Bottleneck dimension (default: 16).
            dropout_prob: Dropout probability for MC Dropout uncertainty calculation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob

        # Encoder: input_dim -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, latent_dim)
        )

        # Decoder: 16 -> 32 -> 64 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for reconstruction.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim).
        """
        # If input is 3D sequence (batch_size, seq_len, input_dim), flatten or use latest step
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x_flat = x.view(-1, dim)
            latent = self.encoder(x_flat)
            recon_flat = self.decoder(latent)
            return recon_flat.view(batch_size, seq_len, dim)
        
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

    def get_novelty_score(self, x: torch.Tensor) -> np.ndarray:
        """
        Computes per-sample Mean Squared Error (MSE) reconstruction error,
        scaled into a 0.0 to 1.0 novelty score.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim).

        Returns:
            np.ndarray: 1D array of shape (batch_size,) containing normalized novelty scores.
        """
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            if x.dim() == 3:
                mse = torch.mean((x - x_hat) ** 2, dim=(1, 2)).cpu().numpy()
            else:
                mse = torch.mean((x - x_hat) ** 2, dim=1).cpu().numpy()

        scaled_score = np.clip(mse / (mse.max() + 1e-6), 0.0, 1.0)
        return scaled_score

    def get_confidence_interval(
        self,
        x: torch.Tensor,
        n_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates reconstruction uncertainty using Monte Carlo (MC) Dropout across multiple forward passes.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or 3D sequence.
            n_samples: Number of MC stochastic forward passes (default: 10).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - mean_recon: Mean reconstruction tensor across MC iterations
                - lower_bound: 5th percentile lower bound confidence limit
                - upper_bound: 95th percentile upper bound confidence limit
        """
        # Enable dropout layers explicitly during evaluation for MC sampling
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        recons = []
        with torch.no_grad():
            for _ in range(n_samples):
                x_hat = self.forward(x)
                recons.append(x_hat.cpu().numpy())

        recons_arr = np.array(recons)
        mean_recon = np.mean(recons_arr, axis=0)
        lower_bound = np.percentile(recons_arr, 5, axis=0)
        upper_bound = np.percentile(recons_arr, 95, axis=0)

        self.eval()
        return mean_recon, lower_bound, upper_bound
