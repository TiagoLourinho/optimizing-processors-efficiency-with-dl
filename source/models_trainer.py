import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from models.dataset import CustomDataset
from models.standardizer import Standardizer
from models.predictor import NVMLMetricsPredictor
from models.ptx_encoder import PTXEncoder
from my_lib.utils import collect_system_info
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def forward_batch(
    config: dict,
    batch: dict,
    device,
    ptx_encoder: PTXEncoder,
    power_predictor: NVMLMetricsPredictor,
    runtime_predictor: NVMLMetricsPredictor,
    standardizer: Standardizer,
    criterion,
):
    split_ptx = batch["split_ptx"]
    core_freq = batch["graphics_frequency"].to(device)
    mem_freq = batch["memory_frequency"].to(device)
    ncu_metrics = batch["ncu_metrics"].to(device)
    power_gold = batch["power_gold"].to(device)
    runtime_gold = batch["runtime_gold"].to(device)

    categorical_parts = [t.to(device) for t in split_ptx["categorical_kernels_parts"]]
    numerical_parts = [t.to(device) for t in split_ptx["numerical_kernels_parts"]]
    ptx_vec = ptx_encoder(categorical_parts, numerical_parts)

    power_pred = power_predictor(ptx_vec, core_freq, mem_freq, ncu_metrics)
    runtime_pred = runtime_predictor(ptx_vec, core_freq, mem_freq, ncu_metrics)

    power_loss = criterion(power_pred, power_gold)
    runtime_loss = criterion(runtime_pred, runtime_gold)

    loss = (
        config["power_loss_weight"] * power_loss
        + config["runtime_loss_weight"] * runtime_loss
    )

    power_gold, runtime_gold = standardizer.inv_transform_targets(
        power=power_gold, runtime=runtime_gold
    )
    power_pred, runtime_pred = standardizer.inv_transform_targets(
        power=power_pred, runtime=runtime_pred
    )

    return loss, power_pred, runtime_pred, power_gold, runtime_gold


def main(config: dict):
    # Set random seed for reproducibility
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Prevents cuDNN from auto-tuning convolution algorithms based on performance heuristics (which introduces randomness)
    np.random.seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    system_info = collect_system_info(gpu_name=gpu_name)

    # Loading data
    with open("training_data.json", "r") as f:
        data = json.load(f)

    categorical_sizes = data["models_info"]["categorical_sizes"]
    n_ncu_metrics = data["models_info"]["n_ncu_metrics"]
    ptx_n_numerical_features = data["models_info"]["n_numerical_features"]
    all_samples = data["training_data"]
    all_ptx = data["ptxs"]

    # Split into train and test
    dataset_size = len(all_samples)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_index = int(config["train_percent"] * dataset_size)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_samples = [all_samples[i] for i in train_indices]
    test_samples = [all_samples[i] for i in test_indices]

    # Standardize data
    standardizer = Standardizer(ptx_n_numerical_features=ptx_n_numerical_features)
    standardizer.fit(train_samples=train_samples, all_ptx=all_ptx)
    standardizer.transform_ptx(all_ptx=all_ptx)
    standardizer.transform_samples(samples=train_samples)
    standardizer.transform_samples(samples=test_samples)

    train_loader = DataLoader(
        CustomDataset(samples=train_samples, all_ptx=all_ptx),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        CustomDataset(samples=test_samples, all_ptx=all_ptx),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # Initialize models
    ptx_encoder = PTXEncoder(
        vocab_sizes=categorical_sizes,
        embedding_dim=config["categorical_embedding_dim"],
        numerical_dim=ptx_n_numerical_features,
        hidden_dim=config["lstm_hidden_dim"],
        num_layers=config["lstm_layers"],
        dropout_prob=config["dropout_rate"],
    ).to(device)

    power_predictor = NVMLMetricsPredictor(
        ptx_dim=config["lstm_hidden_dim"],
        ncu_dim=n_ncu_metrics,
        number_of_layers=config["fnn_layers"],
        hidden_dim=config["fnn_hidden_dim"],
        dropout_rate=config["dropout_rate"],
        use_ncu_metrics=config["use_ncu_metrics"],
    ).to(device)

    runtime_predictor = NVMLMetricsPredictor(
        ptx_dim=config["lstm_hidden_dim"],
        ncu_dim=n_ncu_metrics,
        number_of_layers=config["fnn_layers"],
        hidden_dim=config["fnn_hidden_dim"],
        dropout_rate=config["dropout_rate"],
        use_ncu_metrics=config["use_ncu_metrics"],
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
    test_r2_values = []

    # To restore best model states
    best_test_r2 = float("-inf")
    best_epoch = None
    best_ptx_encoder_state = None
    best_power_predictor_state = None
    best_runtime_predictor_state = None

    try:
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

                loss, *_ = forward_batch(
                    config=config,
                    batch=batch,
                    device=device,
                    ptx_encoder=ptx_encoder,
                    power_predictor=power_predictor,
                    runtime_predictor=runtime_predictor,
                    standardizer=standardizer,
                    criterion=criterion,
                )
                train_loss += loss.item()

                loss.backward()

                # Clip gradients to prevent exploding gradients that cause ValueError: Input contains NaN.
                nn.utils.clip_grad_norm_(
                    ptx_encoder.parameters(), config["max_grad_norm"]
                )
                nn.utils.clip_grad_norm_(
                    runtime_predictor.parameters(), config["max_grad_norm"]
                )
                nn.utils.clip_grad_norm_(
                    power_predictor.parameters(), config["max_grad_norm"]
                )

                ptx_optimizer.step()
                power_optimizer.step()
                runtime_optimizer.step()

            train_avg_loss = train_loss / len(train_loader)

            ########## Evaluate on test set ##########

            ptx_encoder.eval()
            power_predictor.eval()
            runtime_predictor.eval()

            test_loss = 0

            # For R² calculation
            all_power_preds, all_runtime_preds = [], []
            all_power_golds, all_runtime_golds = [], []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing", leave=False):

                    loss, power_pred, runtime_pred, power_gold, runtime_gold = (
                        forward_batch(
                            config=config,
                            batch=batch,
                            device=device,
                            ptx_encoder=ptx_encoder,
                            power_predictor=power_predictor,
                            runtime_predictor=runtime_predictor,
                            standardizer=standardizer,
                            criterion=criterion,
                        )
                    )

                    test_loss += loss.item()

                    all_power_preds.append(power_pred)
                    all_runtime_preds.append(runtime_pred)
                    all_power_golds.append(power_gold)
                    all_runtime_golds.append(runtime_gold)

            test_avg_loss = test_loss / len(test_loader)
            test_r2 = (
                r2_score(all_power_golds, all_power_preds)
                + r2_score(all_runtime_golds, all_runtime_preds)
            ) / 2

            # Store values for the results JSON
            train_loss_values.append(train_avg_loss)
            test_loss_values.append(test_avg_loss)
            test_r2_values.append(test_r2)

            print(
                f"Average losses: Train: {train_avg_loss:.4f}\t Test: {test_avg_loss:.4f}"
            )
            print(f"Test R² score: {test_r2}")

            # Save best performing epoch state
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_epoch = epoch
                best_ptx_encoder_state = copy.deepcopy(ptx_encoder.state_dict())
                best_power_predictor_state = copy.deepcopy(power_predictor.state_dict())
                best_runtime_predictor_state = copy.deepcopy(
                    runtime_predictor.state_dict()
                )
    except (Exception, KeyboardInterrupt) as e:
        print(f"Stopping training early due to an error: {e}")

    # Save models
    torch.save(best_ptx_encoder_state, "ptx_encoder.pth")
    torch.save(best_power_predictor_state, "power_predictor.pth")
    torch.save(best_runtime_predictor_state, "runtime_predictor.pth")

    print(f"\nModels from best epoch {best_epoch+1} saved successfully!")

    ########## Restore best weights and get final predictions ##########

    ptx_encoder.load_state_dict(best_ptx_encoder_state)
    power_predictor.load_state_dict(best_power_predictor_state)
    runtime_predictor.load_state_dict(best_runtime_predictor_state)

    ptx_encoder.eval()
    power_predictor.eval()
    runtime_predictor.eval()

    all_predictions = {"train": {}, "test": {}}
    with torch.no_grad():
        for split_key in all_predictions:

            dataloader = train_loader if split_key == "train" else test_loader

            for batch in tqdm(
                dataloader,
                desc=f"Getting final predicitons for {split_key} set",
                leave=False,
            ):
                _, power_pred, runtime_pred, power_gold, runtime_gold = forward_batch(
                    config=config,
                    batch=batch,
                    device=device,
                    ptx_encoder=ptx_encoder,
                    power_predictor=power_predictor,
                    runtime_predictor=runtime_predictor,
                    standardizer=standardizer,
                    criterion=criterion,
                )

                all_predictions[split_key][
                    f'{batch["benchmark_name"]}_at_f_mem={int(batch["memory_frequency"].item())}_f_core={int(batch["graphics_frequency"].item())}'
                ] = {
                    "runtime_predict": runtime_pred,
                    "runtime_gold": runtime_gold,
                    "power_predict": power_pred,
                    "power_gold": power_gold,
                }

    ########## Save results summary ##########

    results_summary = "models_training_summary.json"
    with open(results_summary, "w") as json_file:
        summary = {
            "config": config,
            "system_info": system_info,
            "best_epoch_results": {
                "best_epoch": best_epoch,
                "best_train_loss": train_loss_values[best_epoch],
                "best_test_loss": test_loss_values[best_epoch],
                "best_test_r2": test_r2_values[best_epoch],
            },
            "values_per_epoch": {
                "train_loss": train_loss_values,
                "test_loss": test_loss_values,
                "test_r2": test_r2_values,
            },
            "all_best_predictions": all_predictions,
        }
        json.dump(summary, json_file, indent=4)
        print(f"\nExported results summary to {results_summary}.")


if __name__ == "__main__":
    config["models_trainer"].update(
        {
            "memory_levels": {
                "core": config["benchmarks_to_training_data"]["n_closest_core_clocks"],
                "mem": config["benchmarks_to_training_data"]["n_closest_mem_clocks"],
            }
        }
    )
    main(config=config["models_trainer"])
