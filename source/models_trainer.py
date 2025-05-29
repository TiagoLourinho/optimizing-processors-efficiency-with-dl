import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from models.dataset import CustomDataset
from models.predictor import FrequencyScalingPredictor
from models.ptx_encoder import PTXEncoder
from models.standardizer import Standardizer
from my_lib.utils import collect_system_info
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def add_targets(samples: list) -> None:
    """Adds the targets to the samples list (in place)"""

    # ED²P = E * D² = P * D * D² = P * D³
    get_ed2p = lambda power, runtime: power * (runtime**3)

    get_scaling_factor = lambda real_freq, optimal_freq: optimal_freq / real_freq

    # Find mininum ED²P for each benchmark
    min_ed2p_per_benchmark = {}
    optimal_freqs_per_benchmark = {}
    for sample in samples:
        benchmark_name = sample["benchmark_name"]
        if benchmark_name not in min_ed2p_per_benchmark:
            min_ed2p_per_benchmark[benchmark_name] = float("inf")
            optimal_freqs_per_benchmark[benchmark_name] = {}

        ed2p_value = get_ed2p(
            sample["benchmark_metrics"]["nvml_metrics"]["POWER"],
            sample["benchmark_metrics"]["runtime"],
        )
        if ed2p_value < min_ed2p_per_benchmark[benchmark_name]:
            min_ed2p_per_benchmark[benchmark_name] = ed2p_value
            optimal_freqs_per_benchmark[benchmark_name]["memory_frequency"] = sample[
                "memory_frequency"
            ]
            optimal_freqs_per_benchmark[benchmark_name]["graphics_frequency"] = sample[
                "graphics_frequency"
            ]

    # Add targets to samples
    for sample in samples:
        benchmark_name = sample["benchmark_name"]
        optimal_freqs = optimal_freqs_per_benchmark[benchmark_name]

        sample["targets"] = {
            "memory_scaling_factor": get_scaling_factor(
                sample["memory_frequency"], optimal_freqs["memory_frequency"]
            ),
            "graphics_scaling_factor": get_scaling_factor(
                sample["graphics_frequency"], optimal_freqs["graphics_frequency"]
            ),
        }


def forward_batch(
    batch: dict,
    device,
    ptx_encoder: PTXEncoder,
    memory_predictor: FrequencyScalingPredictor,
    graphics_predictor: FrequencyScalingPredictor,
    standardizer: Standardizer,
    criterion,
):
    split_ptx = batch["split_ptx"]
    core_freq = batch["graphics_frequency"].to(device)
    mem_freq = batch["memory_frequency"].to(device)
    nvml_metrics = batch["nvml_metrics"].to(device)
    cupti_metrics = batch["cupti_metrics"].to(device)
    memory_gold = batch["targets"]["memory_scaling_factor"].to(device)
    graphics_gold = batch["targets"]["graphics_scaling_factor"].to(device)

    categorical_parts = [t.to(device) for t in split_ptx["categorical_kernels_parts"]]
    numerical_parts = [t.to(device) for t in split_ptx["numerical_kernels_parts"]]
    ptx_vec = ptx_encoder(categorical_parts, numerical_parts)

    memory_pred = memory_predictor(
        ptx_vec, core_freq, mem_freq, nvml_metrics, cupti_metrics
    )
    graphics_pred = graphics_predictor(
        ptx_vec, core_freq, mem_freq, nvml_metrics, cupti_metrics
    )

    memory_loss = criterion(memory_pred, memory_gold)
    graphics_loss = criterion(graphics_pred, graphics_gold)

    loss = memory_loss + graphics_loss

    memory_gold, graphics_gold = standardizer.inv_transform_targets(
        memory_scaling_factor=memory_gold, graphics_scaling_factor=graphics_gold
    )
    memory_pred, graphics_pred = standardizer.inv_transform_targets(
        memory_scaling_factor=memory_pred, graphics_scaling_factor=graphics_pred
    )

    return loss, memory_pred, graphics_pred, memory_gold, graphics_gold


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
    n_nvml_metrics = data["models_info"]["n_nvml_metrics"]
    n_cupti_metrics = data["models_info"]["n_cupti_metrics"]
    ptx_n_numerical_features = data["models_info"]["n_numerical_features"]
    all_samples = data["training_data"]
    all_ptx = data["ptxs"]

    # Calculate targets
    add_targets(all_samples)

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

    memory_predictor = FrequencyScalingPredictor(
        ptx_dim=config["lstm_hidden_dim"],
        nvml_dim=n_nvml_metrics,
        cupti_dim=n_cupti_metrics,
        number_of_layers=config["fnn_layers"],
        hidden_dim=config["fnn_hidden_dim"],
        dropout_rate=config["dropout_rate"],
        use_dynamic_info=config["use_dynamic_info"],
    ).to(device)

    graphics_predictor = FrequencyScalingPredictor(
        ptx_dim=config["lstm_hidden_dim"],
        nvml_dim=n_nvml_metrics,
        cupti_dim=n_cupti_metrics,
        number_of_layers=config["fnn_layers"],
        hidden_dim=config["fnn_hidden_dim"],
        dropout_rate=config["dropout_rate"],
        use_dynamic_info=config["use_dynamic_info"],
    ).to(device)

    models_parameters = {
        "ptx_encoder": {
            "vocab_sizes": categorical_sizes,
            "embedding_dim": config["categorical_embedding_dim"],
            "numerical_dim": ptx_n_numerical_features,
            "hidden_dim": config["lstm_hidden_dim"],
            "num_layers": config["lstm_layers"],
            "dropout_prob": config["dropout_rate"],
        },
        "predictors": {
            "ptx_dim": config["lstm_hidden_dim"],
            "nvml_dim": n_nvml_metrics,
            "cupti_dim": n_cupti_metrics,
            "number_of_layers": config["fnn_layers"],
            "hidden_dim": config["fnn_hidden_dim"],
            "dropout_rate": config["dropout_rate"],
            "use_dynamic_info": config["use_dynamic_info"],
        },
    }

    # Optimizers and loss function
    ptx_optimizer = optim.Adam(ptx_encoder.parameters(), lr=config["learning_rate"])
    memory_optimizer = optim.Adam(
        memory_predictor.parameters(), lr=config["learning_rate"]
    )
    graphics_optimizer = optim.Adam(
        graphics_predictor.parameters(), lr=config["learning_rate"]
    )
    criterion = nn.MSELoss()

    # Store info for the results JSON
    train_loss_values = []
    test_loss_values = []
    test_r2_values = []

    # To restore best model states
    best_test_r2 = float("-inf")
    best_epoch = -1
    best_ptx_encoder_state = copy.deepcopy(ptx_encoder.state_dict())
    best_memory_predictor_state = copy.deepcopy(memory_predictor.state_dict())
    best_graphics_predictor_state = copy.deepcopy(graphics_predictor.state_dict())

    try:
        for epoch in range(config["epochs"]):
            print(f'Epoch {epoch + 1}/{config["epochs"]}')

            ########## Train ##########

            ptx_encoder.train()
            memory_predictor.train()
            graphics_predictor.train()

            train_loss = 0

            for batch in tqdm(train_loader, desc="Training", leave=False):
                graphics_optimizer.zero_grad()
                memory_optimizer.zero_grad()
                ptx_optimizer.zero_grad()

                loss, *_ = forward_batch(
                    batch=batch,
                    device=device,
                    ptx_encoder=ptx_encoder,
                    memory_predictor=memory_predictor,
                    graphics_predictor=graphics_predictor,
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
                    graphics_predictor.parameters(), config["max_grad_norm"]
                )
                nn.utils.clip_grad_norm_(
                    memory_predictor.parameters(), config["max_grad_norm"]
                )

                ptx_optimizer.step()
                memory_optimizer.step()
                graphics_optimizer.step()

            train_avg_loss = train_loss / len(train_loader)

            ########## Evaluate on test set ##########

            ptx_encoder.eval()
            memory_predictor.eval()
            graphics_predictor.eval()

            test_loss = 0

            # For R² calculation
            all_memory_preds, all_graphics_preds = [], []
            all_memory_golds, all_graphics_golds = [], []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing", leave=False):

                    loss, memory_pred, graphics_pred, memory_gold, graphics_gold = (
                        forward_batch(
                            batch=batch,
                            device=device,
                            ptx_encoder=ptx_encoder,
                            memory_predictor=memory_predictor,
                            graphics_predictor=graphics_predictor,
                            standardizer=standardizer,
                            criterion=criterion,
                        )
                    )

                    test_loss += loss.item()

                    all_memory_preds.append(memory_pred)
                    all_graphics_preds.append(graphics_pred)
                    all_memory_golds.append(memory_gold)
                    all_graphics_golds.append(graphics_gold)

            test_avg_loss = test_loss / len(test_loader)
            test_r2 = (
                r2_score(all_memory_golds, all_memory_preds)
                + r2_score(all_graphics_golds, all_graphics_preds)
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
                best_memory_predictor_state = copy.deepcopy(
                    memory_predictor.state_dict()
                )
                best_graphics_predictor_state = copy.deepcopy(
                    graphics_predictor.state_dict()
                )
    except (Exception, KeyboardInterrupt) as e:
        print(f"Stopping training early due to an error: {str(e)}")

    # Save models
    torch.save(best_ptx_encoder_state, "ptx_encoder.pth")
    torch.save(best_memory_predictor_state, "memory_predictor.pth")
    torch.save(best_graphics_predictor_state, "graphics_predictor.pth")
    standardizer.save("standardizer.pkl")
    with open("models_parameters.json", "w") as f:
        json.dump(models_parameters, f, indent=4)

    print(f"\nModels from best epoch {best_epoch+1} saved successfully!")

    ########## Restore best weights and get final predictions ##########

    ptx_encoder.load_state_dict(best_ptx_encoder_state)
    memory_predictor.load_state_dict(best_memory_predictor_state)
    graphics_predictor.load_state_dict(best_graphics_predictor_state)

    ptx_encoder.eval()
    memory_predictor.eval()
    graphics_predictor.eval()

    all_predictions = {"train": {}, "test": {}}
    with torch.no_grad():
        for split_key in all_predictions:

            dataloader = train_loader if split_key == "train" else test_loader

            for batch in tqdm(
                dataloader,
                desc=f"Getting final predicitons for {split_key} set",
                leave=False,
            ):
                _, memory_pred, graphics_pred, memory_gold, graphics_gold = (
                    forward_batch(
                        batch=batch,
                        device=device,
                        ptx_encoder=ptx_encoder,
                        memory_predictor=memory_predictor,
                        graphics_predictor=graphics_predictor,
                        standardizer=standardizer,
                        criterion=criterion,
                    )
                )

                all_predictions[split_key][
                    f'{batch["benchmark_name"]}_at_f_mem={int(batch["memory_frequency"].item())}_f_core={int(batch["graphics_frequency"].item())}'
                ] = {
                    "graphics_pred": graphics_pred,
                    "graphics_gold": graphics_gold,
                    "memory_pred": memory_pred,
                    "memory_gold": memory_gold,
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
            "frequency_levels": {
                "core": config["benchmarks_to_training_data"]["n_closest_core_clocks"],
                "mem": config["benchmarks_to_training_data"]["n_closest_mem_clocks"],
            }
        }
    )
    main(config=config["models_trainer"])
