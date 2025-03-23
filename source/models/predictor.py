import torch
import torch.nn as nn
import torch.nn.functional as F


class NVMLScalingFactorsPredictor(nn.Module):
    """
    Neural network model for predicting an NVML-related performance metric scaling factor from
    PTX embeddings, operating frequencies and NCU features.

    Parameters
    ----------
    ptx_dim : int
        Dimension of the PTX vector representation.
    ncu_dim : int
        Number of NCU-related metrics.
    hidden_dim : int
        Number of hidden units in fully connected layers.
    dropout_rate : float
        Dropout rate for regularization.
    """

    def __init__(
        self, ptx_dim: int, ncu_dim: int, hidden_dim: int, dropout_rate: float
    ):
        super(NVMLScalingFactorsPredictor, self).__init__()

        # FC layer for PTX vector representation
        self.fc_ptx = nn.Linear(ptx_dim, hidden_dim)

        # FC layer for hardware performance metrics (core freq, mem freq, and NCU metrics)
        self.fc_others = nn.Linear(
            2 + ncu_dim, hidden_dim
        )  # 2 from core & mem freq, plus NCU metrics

        # Fully connected layers for feature fusion
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Only 1 metric is predicted each time

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        ptx_vec: torch.Tensor,
        core_freq: torch.Tensor,
        mem_freq: torch.Tensor,
        ncu_metrics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        ptx_vec : torch.Tensor
            torch.Tensor of shape (ptx_dim,) representing the PTX vector.
        core_freq : torch.Tensor
            torch.Tensor of shape (1,) representing core frequency.
        mem_freq : torch.Tensor
            torch.Tensor of shape (1,) representing memory frequency.
        ncu_metrics : torch.Tensor
            torch.Tensor of shape (ncu_dim,) representing multiple NCU metrics.

        Returns
        -------
        torch.Tensor
            Predicted NVML-related scaling factor of shape (1,).
        """
        # Process PTX embedding
        ptx_out = F.relu(self.fc_ptx(ptx_vec))

        # Process hardware metrics
        other_metrics = torch.cat([core_freq, mem_freq, ncu_metrics], dim=0)
        other_metrics_out = F.relu(self.fc_others(other_metrics))

        # Concatenate features
        combined_features = torch.cat([ptx_out, other_metrics_out], dim=0)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
