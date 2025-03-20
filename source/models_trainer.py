import json

import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import TrainingDataset
from models.predictor import NVMLMetricsPredictor
from models.ptx_encoder import PTXEncoder
from torch.utils.data import DataLoader


def main():

    # Loading data
    with open("training_data.json", "r") as f:
        data = json.load(f)
    dataset = TrainingDataset(samples=data["training_data"], ptx=data["ptxs"])
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

    # Hyperparameters
    categorical_sizes = data["models_info"]["categorical_sizes"]
    n_ncu_metrics = data["models_info"]["n_ncu_metrics"]
    n_numerical_features = data["models_info"]["n_numerical_features"]
    categorical_embedding_dim = 16
    lstm_hidden_dim = 128
    lstm_layers = 1
    fnn_hidden_dim = 128
    dropout_rate = 0.3
    learning_rate = 0.001
    epochs = 10
    runtime_loss_weight = 1
    power_loss_weight = 1

    # Initialize models
    ptx_encoder = PTXEncoder(
        vocab_sizes=categorical_sizes,
        embedding_dim=categorical_embedding_dim,
        numerical_dim=n_numerical_features,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_layers,
    )
    power_predictor = NVMLMetricsPredictor(
        ptx_dim=lstm_hidden_dim,
        ncu_dim=n_ncu_metrics,
        hidden_dim=fnn_hidden_dim,
        dropout_rate=dropout_rate,
    )
    runtime_predictor = NVMLMetricsPredictor(
        ptx_dim=lstm_hidden_dim,
        ncu_dim=n_ncu_metrics,
        hidden_dim=fnn_hidden_dim,
        dropout_rate=dropout_rate,
    )

    # Optimizers and loss function
    ptx_optimizer = optim.Adam(ptx_encoder.parameters(), lr=learning_rate)
    power_optimizer = optim.Adam(power_predictor.parameters(), lr=learning_rate)
    runtime_optimizer = optim.Adam(runtime_predictor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            split_ptx = batch["split_ptx"]
            core_freq = batch["graphics_frequency"]
            mem_freq = batch["memory_frequency"]
            ncu_metrics = batch["ncu_metrics"]
            avg_power = batch["average_POWER"]
            avg_runtime = batch["median_run_time"]

            # Reset
            runtime_optimizer.zero_grad()
            power_optimizer.zero_grad()
            ptx_optimizer.zero_grad()

            # Encoding
            ptx_vec = ptx_encoder(
                split_ptx["categorical_kernels_parts"],
                split_ptx["numerical_kernels_parts"],
            )

            # Predict power
            power_prediction = power_predictor(
                ptx_vec, core_freq, mem_freq, ncu_metrics
            )
            power_loss = criterion(power_prediction, avg_power)

            # Predict runtime
            runtime_prediction = runtime_predictor(
                ptx_vec, core_freq, mem_freq, ncu_metrics
            )
            runtime_loss = criterion(runtime_prediction, avg_runtime)

            # Total loss calculation
            loss = power_loss_weight * power_loss + runtime_loss_weight * runtime_loss
            total_loss += loss.item()

            # Backpropagate
            loss.backward()
            ptx_optimizer.step()
            power_optimizer.step()
            runtime_optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save trained models
    torch.save(ptx_encoder.state_dict(), "ptx_encoder.pth")
    torch.save(power_predictor.state_dict(), "power_predictor.pth")
    torch.save(runtime_predictor.state_dict(), "runtime_predictor.pth")

    print("\nModels saved successfully!")


if __name__ == "__main__":
    main()
