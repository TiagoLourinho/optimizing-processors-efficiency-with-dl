import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from models.dataset import TrainingDataset
from models.predictor import NVMLScalingFactorsPredictor
from models.ptx_encoder import PTXEncoder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def forward_batch(
    config: dict,
    batch: dict,
    device,
    ptx_encoder: PTXEncoder,
    power_predictor: NVMLScalingFactorsPredictor,
    runtime_predictor: NVMLScalingFactorsPredictor,
    criterion,
):
    split_ptx = batch["split_ptx"]
    core_freq = batch["graphics_frequency"].to(device)
    mem_freq = batch["memory_frequency"].to(device)
    ncu_metrics = batch["ncu_metrics"].to(device)
    power_gold = batch["power_scaling_factor"].to(device)
    runtime_gold = batch["runtime_scaling_factor"].to(device)

    categorical_parts = [t.to(device) for t in split_ptx["categorical_kernels_parts"]]
    numerical_parts = [t.to(device) for t in split_ptx["numerical_kernels_parts"]]
    ptx_vec = ptx_encoder(categorical_parts, numerical_parts)

    power_pred = power_predictor(ptx_vec, core_freq, mem_freq, ncu_metrics)
    runtime_pred = runtime_predictor(ptx_vec, core_freq, mem_freq, ncu_metrics)

    power_loss = criterion(power_pred, power_gold)
    runtime_loss = criterion(runtime_pred, runtime_gold)

    return (
        config["power_loss_weight"] * power_loss
        + config["runtime_loss_weight"] * runtime_loss
    )


def main(config: dict):
    # Set random seed for reproducibility
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])

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
    split_index = int(config["train_percent"] * dataset_size)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    categorical_sizes = data["models_info"]["categorical_sizes"]
    n_ncu_metrics = data["models_info"]["n_ncu_metrics"]
    n_numerical_features = data["models_info"]["n_numerical_features"]

    # Initialize models
    ptx_encoder = PTXEncoder(
        vocab_sizes=categorical_sizes,
        embedding_dim=config["categorical_embedding_dim"],
        numerical_dim=n_numerical_features,
        hidden_dim=config["lstm_hidden_dim"],
        num_layers=config["lstm_layers"],
    ).to(device)

    power_predictor = NVMLScalingFactorsPredictor(
        ptx_dim=config["lstm_hidden_dim"],
        ncu_dim=n_ncu_metrics,
        hidden_dim=config["fnn_hidden_dim"],
        dropout_rate=config["dropout_rate"],
    ).to(device)

    runtime_predictor = NVMLScalingFactorsPredictor(
        ptx_dim=config["lstm_hidden_dim"],
        ncu_dim=n_ncu_metrics,
        hidden_dim=config["fnn_hidden_dim"],
        dropout_rate=config["dropout_rate"],
    ).to(device)

    # Optimizers and loss function
    ptx_optimizer = optim.Adam(ptx_encoder.parameters(), lr=config["learning_rate"])
    power_optimizer = optim.Adam(
        power_predictor.parameters(), lr=config["learning_rate"]
    )
    runtime_optimizer = optim.Adam(
        runtime_predictor.parameters(), lr=config["learning_rate"]
    )
    criterion = nn.MSELoss()

    # Store info for the results JSON
    train_loss_values = []
    test_loss_values = []

    for epoch in range(config["epochs"]):
        print(f'Epoch {epoch + 1}/{config["epochs"]}')

        ########## Train ##########

        ptx_encoder.train()
        power_predictor.train()
        runtime_predictor.train()

        train_loss = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            runtime_optimizer.zero_grad()
            power_optimizer.zero_grad()
            ptx_optimizer.zero_grad()

            loss = forward_batch(
                config=config,
                batch=batch,
                device=device,
                ptx_encoder=ptx_encoder,
                power_predictor=power_predictor,
                runtime_predictor=runtime_predictor,
                criterion=criterion,
            )
            train_loss += loss.item()

            loss.backward()
            ptx_optimizer.step()
            power_optimizer.step()
            runtime_optimizer.step()

        train_avg_loss = train_loss / len(train_loader)
        train_loss_values.append(train_avg_loss)

        ########## Evaluate on test set ##########

        ptx_encoder.eval()
        power_predictor.eval()
        runtime_predictor.eval()

        test_loss = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", leave=False):
                test_loss += forward_batch(
                    config=config,
                    batch=batch,
                    device=device,
                    ptx_encoder=ptx_encoder,
                    power_predictor=power_predictor,
                    runtime_predictor=runtime_predictor,
                    criterion=criterion,
                ).item()

        test_avg_loss = test_loss / len(test_loader)
        test_loss_values.append(test_avg_loss)

        print(
            f"Average losses:\nTrain: {train_avg_loss:.4f}\t Test: {test_avg_loss:.4f}"
        )

    # Save models
    torch.save(ptx_encoder.state_dict(), "ptx_encoder.pth")
    torch.save(power_predictor.state_dict(), "power_predictor.pth")
    torch.save(runtime_predictor.state_dict(), "runtime_predictor.pth")

    print("\nModels saved successfully!")


if __name__ == "__main__":
    main(config=config["models_trainer"])
