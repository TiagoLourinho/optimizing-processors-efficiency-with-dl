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
    number_of_layers : int
        Number of fully connected layers.
    hidden_dim : int
        Number of hidden units in fully connected layers.
    dropout_rate : float
        Dropout rate for regularization.
    use_ncu_metrics : bool
        Whether to use NCU metrics or not (runtime information).
    """

    def __init__(
        self,
        ptx_dim: int,
        ncu_dim: int,
        number_of_layers: int,
        hidden_dim: int,
        dropout_rate: float,
        use_ncu_metrics: bool,
    ):
        super(NVMLScalingFactorsPredictor, self).__init__()

        # PTX, plus 2 from core & mem freq, plus NCU metrics
        self.use_ncu_metrics = use_ncu_metrics
        if use_ncu_metrics:
            input_dim = ptx_dim + 2 + ncu_dim
        else:
            input_dim = ptx_dim + 2

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = torch.nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(number_of_layers)]
        )

        self.relu = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, 1)  # Only 1 metric is predicted each time

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

        all_features = torch.cat([ptx_vec, core_freq, mem_freq], dim=0)

        if self.use_ncu_metrics:
            all_features = torch.cat([all_features, ncu_metrics], dim=0)

        x = self.fc_in(all_features)
        x = self.relu(x)
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.fc_out(x)

        return x
