from typing import List

import torch
import torch.nn as nn


class KernelEncoder(nn.Module):
    """
    Encodes a sequence of encoded instructions into a single vector.

    Parameters
    ----------
    vocab_sizes : List[int]
        List of vocabulary sizes for each categorical feature.
    embedding_dim : int
        Dimension of the embeddings for categorical features.
    numerical_dim : int
        Number of numerical features in each instruction.
    hidden_dim : int
        Hidden dimension of the LSTM.
    num_layers : int
        Number of layers in the LSTM.
    dropout_prob: float
        Dropout probability for the LSTM.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        embedding_dim: int,
        numerical_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_prob: float,
    ):
        super(KernelEncoder, self).__init__()

        # Create embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes]
        )

        self.lstm = nn.LSTM(
            input_size=len(vocab_sizes) * embedding_dim
            + numerical_dim,  # Numerical features are directly concatenated
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
        )

    def forward(
        self, categorical_inputs: torch.Tensor, numerical_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the KernelEncoder.

        Parameters
        ----------
        categorical_inputs : torch.Tensor
            Tensor of shape (seq_len, num_categorical) containing categorical feature indices.
        numerical_inputs : torch.Tensor
            Tensor of shape (seq_len, numerical_dim) containing numerical feature values.

        Returns
        -------
        torch.Tensor
            Encoded kernel vector of shape (hidden_dim,)
        """

        # Embed each categorical feature
        embedded_features = [
            emb(categorical_inputs[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        embedded_features = torch.cat(
            embedded_features, dim=-1
        )  # (seq_len, num_categorical * embedding_dim)

        # Concatenate numerical features directly without transformation
        lstm_input = torch.cat(
            [embedded_features, numerical_inputs], dim=-1
        )  # (seq_len, num_categorical * embedding_dim + numerical_dim)

        _, (hidden_state, _) = self.lstm(lstm_input)

        return hidden_state[-1]  # (hidden_dim,)


class PTXEncoder(nn.Module):
    """
    Encodes a PTX file.

    Parameters
    ----------
    vocab_sizes : List[int]
        List of vocabulary sizes for each categorical feature.
    embedding_dim : int
        Dimension of the embeddings for categorical features.
    numerical_dim : int
        Number of numerical features in each instruction.
    hidden_dim : int
        Hidden dimension of the LSTM.
    num_layers : int
        Number of layers in the LSTM.
    dropout_prob: float
        Dropout probability for the LSTM.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        embedding_dim: int,
        numerical_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_prob: float,
    ) -> None:
        super(PTXEncoder, self).__init__()
        self.kernel_encoder = KernelEncoder(
            vocab_sizes,
            embedding_dim,
            numerical_dim,
            hidden_dim,
            num_layers,
            dropout_prob,
        )

    def forward(
        self,
        categorical_kernels: List[torch.Tensor],
        numerical_kernels: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of the PTXEncoder.

        Parameters
        ----------
        categorical_kernels : List[torch.Tensor]
            List of tensors where each tensor has shape (seq_len, num_categorical),
            representing the categorical features for a kernel.
        numerical_kernels : List[torch.Tensor]
            List of tensors where each tensor has shape (seq_len, numerical_dim),
            representing the numerical features for a kernel.

        Returns
        -------
        torch.Tensor
            Encoded PTX vector of shape (hidden_dim,)
        """
        kernel_vectors = [
            self.kernel_encoder(cat_kernel, num_kernel)
            for cat_kernel, num_kernel in zip(categorical_kernels, numerical_kernels)
        ]

        # Average the kernel representations
        kernel_vectors_tensor = torch.stack(
            kernel_vectors, dim=0
        )  # (num_kernels, hidden_dim)
        ptx_vector = kernel_vectors_tensor.mean(dim=0)  # (hidden_dim,)

        return ptx_vector
