import json
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import TrainingDataset
from models.predictor import NVMLScalingFactorsPredictor
from models.ptx_encoder import PTXEncoder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# Hyperparameters
seed = 1
train_percent = 0.8
batch_size = None

categorical_embedding_dim = 16
lstm_hidden_dim = 128
lstm_layers = 1
fnn_hidden_dim = 128
dropout_rate = 0.3
learning_rate = 0.001
epochs = 100
runtime_loss_weight = 1
power_loss_weight = 1


def main():
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loading data
    with open("training_data.json", "r") as f:
        data = json.load(f)
    full_dataset = TrainingDataset(samples=data["training_data"], ptx=data["ptxs"])

    # Split into train and test
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_index = int(train_percent * dataset_size)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    categorical_sizes = data["models_info"]["categorical_sizes"]
    n_ncu_metrics = data["models_info"]["n_ncu_metrics"]
    n_numerical_features = data["models_info"]["n_numerical_features"]

    # Initialize models
    ptx_encoder = PTXEncoder(
        vocab_sizes=categorical_sizes,
        embedding_dim=categorical_embedding_dim,
        numerical_dim=n_numerical_features,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_layers,
    ).to(device)

    power_predictor = NVMLScalingFactorsPredictor(
        ptx_dim=lstm_hidden_dim,
        ncu_dim=n_ncu_metrics,
        hidden_dim=fnn_hidden_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    runtime_predictor = NVMLScalingFactorsPredictor(
        ptx_dim=lstm_hidden_dim,
        ncu_dim=n_ncu_metrics,
        hidden_dim=fnn_hidden_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    # Optimizers and loss function
    ptx_optimizer = optim.Adam(ptx_encoder.parameters(), lr=learning_rate)
    power_optimizer = optim.Adam(power_predictor.parameters(), lr=learning_rate)
    runtime_optimizer = optim.Adam(runtime_predictor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Plot later
    loss_values = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            benchmark_name = batch["benchmark_name"]
            split_ptx = batch["split_ptx"]

            core_freq = batch["graphics_frequency"].to(device)
            mem_freq = batch["memory_frequency"].to(device)
            ncu_metrics = batch["ncu_metrics"].to(device)
            power_scaling_factor_gold = batch["power_scaling_factor"].to(device)
            runtime_scaling_factor_gold = batch["runtime_scaling_factor"].to(device)

            # Reset
            runtime_optimizer.zero_grad()
            power_optimizer.zero_grad()
            ptx_optimizer.zero_grad()

            # Encode PTX
            categorical_parts = [
                tensor.to(device) for tensor in split_ptx["categorical_kernels_parts"]
            ]
            numerical_parts = [
                tensor.to(device) for tensor in split_ptx["numerical_kernels_parts"]
            ]
            ptx_vec = ptx_encoder(categorical_parts, numerical_parts)

            # Predict and compute loss
            power_scaling_factor_prediction = power_predictor(
                ptx_vec, core_freq, mem_freq, ncu_metrics
            )
            power_loss = criterion(
                power_scaling_factor_prediction, power_scaling_factor_gold
            )

            runtime_scaling_factor_prediction = runtime_predictor(
                ptx_vec, core_freq, mem_freq, ncu_metrics
            )
            runtime_loss = criterion(
                runtime_scaling_factor_prediction, runtime_scaling_factor_gold
            )

            # Total loss
            loss = power_loss_weight * power_loss + runtime_loss_weight * runtime_loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            ptx_optimizer.step()
            power_optimizer.step()
            runtime_optimizer.step()

        average_loss = total_loss / len(train_loader)
        loss_values.append(average_loss)
        print(f"Average Train Loss: {average_loss:.4f}")

    # Save models
    torch.save(ptx_encoder.state_dict(), "ptx_encoder.pth")
    torch.save(power_predictor.state_dict(), "power_predictor.pth")
    torch.save(runtime_predictor.state_dict(), "runtime_predictor.pth")

    print("\nModels saved successfully!")

    # Plot and save loss curve
    plt.figure()
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve.png")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    ptx_encoder.eval()
    power_predictor.eval()
    runtime_predictor.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            split_ptx = batch["split_ptx"]

            core_freq = batch["graphics_frequency"].to(device)
            mem_freq = batch["memory_frequency"].to(device)
            ncu_metrics = batch["ncu_metrics"].to(device)
            power_gold = batch["power_scaling_factor"].to(device)
            runtime_gold = batch["runtime_scaling_factor"].to(device)

            categorical_parts = [
                t.to(device) for t in split_ptx["categorical_kernels_parts"]
            ]
            numerical_parts = [
                t.to(device) for t in split_ptx["numerical_kernels_parts"]
            ]
            ptx_vec = ptx_encoder(categorical_parts, numerical_parts)

            power_pred = power_predictor(ptx_vec, core_freq, mem_freq, ncu_metrics)
            runtime_pred = runtime_predictor(ptx_vec, core_freq, mem_freq, ncu_metrics)

            power_loss = criterion(power_pred, power_gold)
            runtime_loss = criterion(runtime_pred, runtime_gold)

            test_loss += (
                power_loss_weight * power_loss + runtime_loss_weight * runtime_loss
            ).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()
