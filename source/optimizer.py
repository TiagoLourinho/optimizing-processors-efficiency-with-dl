import argparse
import json
import random
import subprocess
import time
from collections import deque

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
from my_lib.utils import closest_value, get_ed2p, get_edp


FREQUENCY_UPDATE_INTERVAL = 0.1  # seconds
CONSECUTIVE_CONTROL_SIGNALS = 2  # Number of consecutive equal control signals to send before actual changing frequencies


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

    # Save values before standardization
    original_memory_freq = gpu.memory_clk
    original_graphics_freq = gpu.graphics_clk

    # Create a dummy sample to use the standardizer
    dummy_samples = [
        {
            "benchmark_name": "application",
            "memory_frequency": original_memory_freq,
            "graphics_frequency": original_graphics_freq,
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

    # Calculate new frequencies using non standardized values
    new_memory_freq = closest_value(
        numbers=gpu.get_supported_memory_clocks(),
        target=original_memory_freq * memory_scaling_factor,
    )
    new_core_freq = closest_value(
        numbers=gpu.get_supported_graphics_clocks(memory_clock=new_memory_freq),
        target=original_graphics_freq * graphics_scaling_factor,
    )

    return new_memory_freq, new_core_freq


def main(
    ptx_file: str,
    executable_file: str,
    executable_args: list[str],
    device: torch.device,
    models_parameters: dict,
    control_clocks: bool,
    plot_control_signals: bool,
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

        if plot_control_signals:
            memory_control_signals = []
            graphics_control_signals = []

    with GPU() as gpu:

        with BenchmarkMonitor(
            gpu=gpu,
            sampling_frequency=None,
            n_runs=None,
        ) as benchmark_monitor:

            # Set to maximum performance at the start
            if control_clocks:
                try:
                    gpu.memory_clk = max(gpu.get_supported_memory_clocks())
                except GPUClockChangingError:
                    print(
                        f"Couldn't set memory clock to maximum, using {gpu.memory_clk} instead."
                    )
                try:
                    gpu.graphics_clk = max(
                        gpu.get_supported_graphics_clocks(memory_clock=gpu.memory_clk)
                    )
                except GPUClockChangingError:
                    print(
                        f"Couldn't set graphics clock to maximum, using {gpu.graphics_clk} instead."
                    )
                time.sleep(1)
                gpu.realtime_mode = True

            print("Starting the application...")

            power_samples = []
            start = time.perf_counter()
            application_process = subprocess.Popen(
                [f"./{executable_file}"] + executable_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            last_memory_freqs = deque(maxlen=CONSECUTIVE_CONTROL_SIGNALS)
            last_graphics_freqs = deque(maxlen=CONSECUTIVE_CONTROL_SIGNALS)
            while True:
                retcode = application_process.poll()

                # Application finished
                if retcode is not None:
                    end = time.perf_counter()
                    final_runtime = end - start
                    final_power = float(np.mean(power_samples))
                    ed2p = get_ed2p(power=final_power, runtime=final_runtime)
                    edp = get_edp(power=final_power, runtime=final_runtime)

                    print(f"\nApplication finished with return code: {retcode}")
                    print(
                        f"Runtime: {final_runtime:.2f} seconds\tAverage power: {final_power:.2f} W"
                    )
                    print(f"EDP: {edp:.2f} J⋅s\tED²P: {ed2p:.2f} J⋅s²")

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
                    last_memory_freqs.append(new_memory_freq)
                    last_graphics_freqs.append(new_core_freq)

                    if plot_control_signals:
                        memory_control_signals.append(new_memory_freq)
                        graphics_control_signals.append(new_core_freq)

                    # Only change frequencies if the last N control signals are the same
                    if len(last_memory_freqs) == CONSECUTIVE_CONTROL_SIGNALS and all(
                        x == last_memory_freqs[0] for x in last_memory_freqs
                    ):
                        gpu.memory_clk = new_memory_freq
                    if len(last_graphics_freqs) == CONSECUTIVE_CONTROL_SIGNALS and all(
                        x == last_graphics_freqs[0] for x in last_graphics_freqs
                    ):
                        gpu.graphics_clk = new_core_freq

            if plot_control_signals:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))
                plt.plot(
                    np.arange(len(memory_control_signals)) * FREQUENCY_UPDATE_INTERVAL,
                    memory_control_signals,
                    label="Memory Clock (MHz)",
                    marker="o",
                )
                plt.plot(
                    np.arange(len(graphics_control_signals))
                    * FREQUENCY_UPDATE_INTERVAL,
                    graphics_control_signals,
                    label="Graphics Clock (MHz)",
                    marker="o",
                )
                plt.xlabel("Time (s)")
                plt.ylabel("Clock Frequency (MHz)")
                plt.title("Control Signals Over Time")
                plt.legend()
                plt.grid()
                plt.savefig("control_signals.png")


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
    parser.add_argument(
        "--plot_control_signals",
        default=False,
        action="store_true",
        help="Plot the control signals over time (default: False)",
    )
    # Everything after `--` is passed to the executable
    parser.add_argument(
        "exec_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the executable (use `--` to separate)",
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
        executable_args=args.exec_args[1:],  # Remove the separator `--`
        device=device,
        models_parameters=models_parameters,
        control_clocks=not args.get_baseline,
        plot_control_signals=args.plot_control_signals,
    )
