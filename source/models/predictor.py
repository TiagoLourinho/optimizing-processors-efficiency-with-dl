import torch
import torch.nn as nn


class NVMLMetricsPredictor(nn.Module):
    """
    Neural network model for predicting an NVML-related performance metrics from
    PTX embeddings, operating frequencies and NSYS features.

    Parameters
    ----------
    ptx_dim : int
        Dimension of the PTX vector representation.
    nsys_dim : int
        Number of NSYS-related metrics.
    number_of_layers : int
        Number of fully connected layers.
    hidden_dim : int
        Number of hidden units in fully connected layers.
    dropout_rate : float
        Dropout rate for regularization.
    use_NSYS_metrics : bool
        Whether to use NSYS metrics or not (runtime information).
    """

    def __init__(
        self,
        ptx_dim: int,
        nsys_dim: int,
        number_of_layers: int,
        hidden_dim: int,
        dropout_rate: float,
        use_nsys_metrics: bool,
    ):
        super(NVMLMetricsPredictor, self).__init__()

        # PTX, plus 2 from core & mem freq, plus NSYS metrics
        self.use_nsys_metrics = use_nsys_metrics
        if use_nsys_metrics:
            input_dim = ptx_dim + 2 + nsys_dim
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
        nsys_metrics: torch.Tensor,
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
        nsys_metrics : torch.Tensor
            torch.Tensor of shape (nsys_dim,) representing multiple NSYS metrics.

        Returns
        -------
        torch.Tensor
            Predicted NVML-related metric of shape (1,).
        """

        all_features = torch.cat([ptx_vec, core_freq, mem_freq], dim=0)

        if self.use_nsys_metrics:
            all_features = torch.cat([all_features, nsys_metrics], dim=0)

        x = self.fc_in(all_features)
        x = self.relu(x)
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.fc_out(x)

        return x
