import argparse
import json
import random
import subprocess
import time

import numpy as np
import torch
from config import config
from models.dataset import CustomDataset
from models.predictor import FrequencyScalingPredictor
from models.ptx_encoder import PTXEncoder
from models.standardizer import Standardizer
from my_lib.PTX_parser import PTXParser
from my_lib.gpu import GPU, GPUClockChangingError
from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.utils import closest_value, get_ed2p


FREQUENCY_UPDATE_INTERVAL = 0.1  # seconds


def get_ptx_embedding(
    ptx_file: str,
    standardizer: Standardizer,
    models_parameters: dict,
    device: torch.device,
) -> torch.Tensor:
    """Creates the embedding of the PTX file."""

    # Load the PTX encoder
    ptx_encoder = PTXEncoder(**models_parameters["ptx_encoder"])
    ptx_encoder.load_state_dict(torch.load("ptx_encoder.pth", weights_only=True))
    ptx_encoder.to(device)
    ptx_encoder.eval()

    # Parse the PTX file
    ptx_parser = PTXParser()
    parsed_ptx = ptx_parser.parse(ptx_file=ptx_file, convert_to_dicts=True)

    # Just to align with the expected input format
    all_ptx = {"application": parsed_ptx}

    # Standardize the PTX data
    standardizer.transform_ptx(all_ptx=all_ptx)

    # Format the PTX data
    split_and_standardized_ptx = CustomDataset.format_ptx(all_ptx)["application"]

    # Create the embedding of the PTX file
    categorical_parts = [
        t.to(device) for t in split_and_standardized_ptx["categorical_kernels_parts"]
    ]
    numerical_parts = [
        t.to(device) for t in split_and_standardized_ptx["numerical_kernels_parts"]
    ]
    ptx_vec = ptx_encoder(categorical_parts, numerical_parts)

    return ptx_vec


def get_new_frequencies(
    gpu: GPU,
    standardizer: Standardizer,
    nvml_metrics: dict,
    cupti_metrics: dict,
    memory_predictor: FrequencyScalingPredictor,
    graphics_predictor: FrequencyScalingPredictor,
    ptx_embedding: torch.Tensor,
    device: torch.device,
):

    # Create a dummy sample to use the standardizer
    dummy_samples = [
        {
            "benchmark_name": "application",
            "memory_frequency": gpu.memory_clk,
            "graphics_frequency": gpu.graphics_clk,
            "benchmark_metrics": {
                "runtime": -1,  # Placeholder for runtime
                "nvml_metrics": nvml_metrics,
                "cupti_metrics": cupti_metrics,
            },
            "targets": {},  # Placeholder for targets
        }
    ]

    # Standardize the dummy sample
    standardizer.transform_samples(samples=dummy_samples)
    sample = dummy_samples[0]

    # Convert data to tensors
    core_freq = (
        torch.tensor(sample["graphics_frequency"], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    mem_freq = (
        torch.tensor(sample["memory_frequency"], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    nvml_metrics = torch.tensor(
        list(sample["benchmark_metrics"]["nvml_metrics"].values()),
        dtype=torch.float32,
    ).to(device)
    cupti_metrics = torch.tensor(
        list(sample["benchmark_metrics"]["cupti_metrics"].values()),
        dtype=torch.float32,
    ).to(device)

    # Get scaling factors
    memory_scaling_factor = memory_predictor(
        ptx_embedding,
        core_freq,
        mem_freq,
        nvml_metrics,
        cupti_metrics,
    )
    graphics_scaling_factor = graphics_predictor(
        ptx_embedding,
        core_freq,
        mem_freq,
        nvml_metrics,
        cupti_metrics,
    )

    # Convert scaling factors to original scale
    memory_scaling_factor, graphics_scaling_factor = standardizer.inv_transform_targets(
        memory_scaling_factor=memory_scaling_factor,
        graphics_scaling_factor=graphics_scaling_factor,
    )

    # Calculate new frequencies
    new_memory_freq = closest_value(
        numbers=gpu.get_supported_memory_clocks(),
        target=float(mem_freq.cpu().item()) * memory_scaling_factor,
    )
    new_core_freq = closest_value(
        numbers=gpu.get_supported_graphics_clocks(memory_clock=new_memory_freq),
        target=float(core_freq.cpu().item()) * graphics_scaling_factor,
    )

    return new_memory_freq, new_core_freq


def main(
    ptx_file: str,
    executable_file: str,
    device: torch.device,
    models_parameters: dict,
    control_clocks: bool,
):
    # Load models
    if control_clocks:
        standardizer = Standardizer.load("standardizer.pkl")
        memory_predictor = FrequencyScalingPredictor(**models_parameters["predictors"])
        memory_predictor.load_state_dict(
            torch.load("memory_predictor.pth", weights_only=True)
        )
        memory_predictor.to(device)
        memory_predictor.eval()
        graphics_predictor = FrequencyScalingPredictor(
            **models_parameters["predictors"]
        )
        graphics_predictor.load_state_dict(
            torch.load("graphics_predictor.pth", weights_only=True)
        )
        graphics_predictor.to(device)
        graphics_predictor.eval()

        # Get the PTX embedding
        ptx_embedding = get_ptx_embedding(
            ptx_file=ptx_file,
            standardizer=standardizer,
            models_parameters=models_parameters,
            device=device,
        )

    with GPU() as gpu:

        with BenchmarkMonitor(
            gpu=gpu,
            sampling_frequency=None,
            n_runs=None,
        ) as benchmark_monitor:

            # Reset to default
            gpu.reset_graphics_clk()
            gpu.reset_memory_clk()
            time.sleep(1)
            gpu.realtime_mode = True

            print("Starting the application...")

            power_samples = []
            start = time.perf_counter()
            application_process = subprocess.Popen(
                [f"./{executable_file}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            while True:
                retcode = application_process.poll()

                # Application finished
                if retcode is not None:
                    end = time.perf_counter()
                    final_runtime = end - start
                    final_power = float(np.mean(power_samples))
                    ed2p = get_ed2p(power=final_power, runtime=final_runtime)

                    print(f"\nApplication finished with return code: {retcode}")
                    print(
                        f"Runtime: {final_runtime:.2f} seconds, Average power: {final_power:.2f} W"
                    )
                    print(f"ED²P: {ed2p:.2f} J/s²\n")

                    break

                time.sleep(FREQUENCY_UPDATE_INTERVAL)

                nvml_metrics = benchmark_monitor.get_nvml_metrics()
                cupti_metrics = benchmark_monitor.get_cupti_metrics()

                power_samples.append(nvml_metrics["POWER"])

                if control_clocks:
                    new_memory_freq, new_core_freq = get_new_frequencies(
                        gpu=gpu,
                        standardizer=standardizer,
                        nvml_metrics=nvml_metrics,
                        cupti_metrics=cupti_metrics,
                        memory_predictor=memory_predictor,
                        graphics_predictor=graphics_predictor,
                        ptx_embedding=ptx_embedding,
                        device=device,
                    )

                    try:
                        gpu.memory_clk = new_memory_freq
                        gpu.graphics_clk = new_core_freq

                        # Sometimes the driver changes the graphics clock after changing the graphics clock
                        assert gpu.memory_clk == new_memory_freq
                        assert gpu.graphics_clk == new_core_freq
                    except (GPUClockChangingError, AssertionError):

                        # The driver sometimes doesn't let the GPU change to frequencies too high so just skip them
                        print(
                            f"\nCouldn't change to memory_clk={new_memory_freq} and graphics_clock={new_core_freq}..."
                        )


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(
        description="Optimizes the ED²P of a running application."
    )
    parser.add_argument(
        "--ptx_path", type=str, required=True, help="Path to the PTX file"
    )
    parser.add_argument(
        "--exec_path", type=str, required=True, help="Path to the executable file"
    )
    parser.add_argument(
        "--get_baseline",
        default=False,
        action="store_true",
        help="Run the benchmark without using the optimizer, clocks are managed by the driver freely (default: False)",
    )
    args = parser.parse_args()

    config = config["models_trainer"]

    # Set random seed for reproducibility
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Prevents cuDNN from auto-tuning convolution algorithms based on performance heuristics (which introduces randomness)
    np.random.seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("models_parameters.json", "r") as f:
        models_parameters = json.load(f)

    main(
        ptx_file=args.ptx_path,
        executable_file=args.exec_path,
        device=device,
        models_parameters=models_parameters,
        control_clocks=not args.get_baseline,
    )
