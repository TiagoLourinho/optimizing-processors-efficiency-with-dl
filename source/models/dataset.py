from typing import List

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset that expects data format exported by the benchmarks_to_training_data.py tool"""

    def __init__(self, samples: List, ptx: dict, models_info: dict):

        # The number of numerical features in the PTX
        self.n_numerical_features = models_info["n_numerical_features"]

        # Standardize before formatting and converting to tensors
        self.__standardize(samples, ptx)

        self.ptxs: dict[str : dict[str:List]] = self.__format_ptx(ptx)
        self.data: List = self.__convert_to_tensors(samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        benchmark_name = sample["benchmark_name"]

        split_ptx = self.ptxs[benchmark_name]

        # Hardware and performance metrics
        core_freq = sample["graphics_frequency"]
        mem_freq = sample["memory_frequency"]
        ncu_metrics = torch.tensor(
            list(sample["ncu_metrics"].values()), dtype=torch.float32
        )  # Remove metrics names from the dictionary

        # Targets
        power_gold = sample["nvml_metrics"]["average_POWER"]
        runtime_gold = sample["nvml_metrics"]["median_run_time"]

        return {
            "benchmark_name": benchmark_name,
            "split_ptx": split_ptx,
            "graphics_frequency": core_freq,
            "memory_frequency": mem_freq,
            "ncu_metrics": ncu_metrics,
            "power_gold": power_gold,
            "runtime_gold": runtime_gold,
        }

    def __convert_to_tensors(self, data):
        """Recursively converts lists of numbers and single numbers in a dictionary to PyTorch tensors."""

        if isinstance(data, dict):
            return {
                key: self.__convert_to_tensors(value) for key, value in data.items()
            }

        elif isinstance(data, list):
            # If fully numeric convert to tensor
            if all(isinstance(x, (int, float)) for x in data):
                return torch.tensor(data, dtype=torch.float32)
            else:
                return [self.__convert_to_tensors(item) for item in data]

        elif isinstance(data, (int, float)):
            return torch.tensor([data], dtype=torch.float32)

        return data

    def __format_ptx(self, ptxs):
        """
        Splits and formats the ptx data into just two lists of equal size, containing the categorical and numerical parts of the kernels

        Example input:
        {
            "cuda-samples_Samples_6_Performance_cudaGraphsPerfScaling_cudaGraphPerfScaling": {
                "_Z5emptyv": [
                    {
                        "categorical": [
                            11,
                            116,
                            0,
                            0,
                            0
                        ],
                        "numerical": [
                            0,
                            0,
                            0,
                            0,
                            0
                        ]
                    }
                ],
                "_Z5delayx": [
                    {
                        "categorical": [
                            8,
                            81,
                            0,
                            8,
                            0
                        ],
                        "numerical": [
                            1,
                            1,
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "categorical": [
                            8,
                            78,
                            0,
                            8,
                            0
                        ],
                        "numerical": [
                            1,
                            1,
                            0,
                            0,
                            0
                        ]
                    }
                ]
        }

        Example output:
        {
            "cuda-samples_Samples_6_Performance_cudaGraphsPerfScaling_cudaGraphPerfScaling": {
                "categorical_kernels_parts": [
                    torch.tensor([
                        [11, 116, 0, 0, 0]
                    ], dtype=torch.int64),  # Shape: (1, 5) for _Z5emptyv
                    torch.tensor([
                        [8, 81, 0, 8, 0],
                        [8, 78, 0, 8, 0]
                    ], dtype=torch.int64)  # Shape: (2, 5) for _Z5delayx
                ],
                "numerical_kernels_parts": [
                    torch.tensor([
                        [0, 0, 0, 0, 0]
                    ], dtype=torch.float32),  # Shape: (1, 5) for _Z5emptyv
                    torch.tensor([
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0]
                    ], dtype=torch.float32)  # Shape: (2, 5) for _Z5delayx
                ]
            }
        }
        """

        formatted_data = {}

        for benchmark_names, kernels in ptxs.items():
            categorical_kernels_parts = []
            numerical_kernels_parts = []

            for kernel_name, kernel_instructions in kernels.items():
                categorical_list = []
                numerical_list = []

                for instruction in kernel_instructions:
                    categorical_list.append(instruction["categorical"])
                    numerical_list.append(instruction["numerical"])

                categorical_tensor = torch.tensor(categorical_list, dtype=torch.int32)
                numerical_tensor = torch.tensor(numerical_list, dtype=torch.float32)

                categorical_kernels_parts.append(categorical_tensor)
                numerical_kernels_parts.append(numerical_tensor)

            formatted_data[benchmark_names] = {
                "categorical_kernels_parts": categorical_kernels_parts,
                "numerical_kernels_parts": numerical_kernels_parts,
            }

        return formatted_data

    def __standardize(self, samples: list[dict], ptxs: dict[str:dict]):
        """Standardizes the data to have 0 mean and 1 std"""

        # Easier to understand while looking at the data format in training_data.json
        #
        # The steps are:
        # 1. Collect all values from the ptx numerical part and the samples
        # 2. Calculate the mean and std for each feature
        # 3. Standardize the values using the formula: (value - mean) / std
        # 4. Replace the original values with the standardized ones

        ######### Initialization #########
        ptx_numerical_part_all_values: list[list[int]] = [
            [] for _ in range(self.n_numerical_features)
        ]
        mem_freq_all_values: list[int] = []
        core_freq_all_values: list[int] = []
        power_all_values: list[float] = []
        runtime_all_values: list[float] = []
        ncu_metrics_all_values: dict[str : list[float]] = (
            {}
        )  # The list for each metric will be later initialized

        ######### All values collection #########
        for benchmark_kernels in ptxs.values():
            for kernel_instructions in benchmark_kernels.values():
                for instruction in kernel_instructions:
                    for dimension in range(self.n_numerical_features):
                        ptx_numerical_part_all_values[dimension].append(
                            instruction["numerical"][dimension]
                        )

        for sample in samples:
            mem_freq_all_values.append(sample["memory_frequency"])
            core_freq_all_values.append(sample["graphics_frequency"])
            power_all_values.append(sample["nvml_metrics"]["average_POWER"])
            runtime_all_values.append(sample["nvml_metrics"]["median_run_time"])

            for metric_name, value in sample["ncu_metrics"].items():
                if metric_name not in ncu_metrics_all_values:
                    ncu_metrics_all_values[metric_name] = []
                ncu_metrics_all_values[metric_name].append(value)

        ######### Convertion to tensors #########
        ptx_numerical_part_all_values = [
            torch.tensor(values, dtype=torch.float32)
            for values in ptx_numerical_part_all_values
        ]
        mem_freq_all_values = torch.tensor(mem_freq_all_values, dtype=torch.float32)
        core_freq_all_values = torch.tensor(core_freq_all_values, dtype=torch.float32)
        power_all_values = torch.tensor(power_all_values, dtype=torch.float32)
        runtime_all_values = torch.tensor(runtime_all_values, dtype=torch.float32)
        ncu_metrics_all_values = {
            metric_name: torch.tensor(values, dtype=torch.float32)
            for metric_name, values in ncu_metrics_all_values.items()
        }

        ######### Means and stds collections #########
        ptx_numerical_part_means_stds = [
            {"mean": values.mean(), "std": values.std()}
            for values in ptx_numerical_part_all_values
        ]
        mem_freq_mean, mem_freq_std = (
            mem_freq_all_values.mean(),
            mem_freq_all_values.std(),
        )
        core_freq_mean, core_freq_std = (
            core_freq_all_values.mean(),
            core_freq_all_values.std(),
        )
        power_mean, power_std = power_all_values.mean(), power_all_values.std()
        runtime_mean, runtime_std = runtime_all_values.mean(), runtime_all_values.std()
        ncu_metrics_means_stds = {
            metric_name: {"mean": values.mean(), "std": values.std()}
            for metric_name, values in ncu_metrics_all_values.items()
        }

        ######### Standardization #########

        standardize = lambda value, mean, std: float((value - mean) / std)

        for benchmark_kernels in ptxs.values():
            for kernel_instructions in benchmark_kernels.values():
                for instruction in kernel_instructions:
                    for dimension in range(self.n_numerical_features):
                        instruction["numerical"][dimension] = standardize(
                            instruction["numerical"][dimension],
                            ptx_numerical_part_means_stds[dimension]["mean"],
                            ptx_numerical_part_means_stds[dimension]["std"],
                        )

        for sample in samples:
            sample["memory_frequency"] = standardize(
                sample["memory_frequency"],
                mem_freq_mean,
                mem_freq_std,
            )
            sample["graphics_frequency"] = standardize(
                sample["graphics_frequency"],
                core_freq_mean,
                core_freq_std,
            )
            sample["nvml_metrics"]["average_POWER"] = standardize(
                sample["nvml_metrics"]["average_POWER"],
                power_mean,
                power_std,
            )
            sample["nvml_metrics"]["median_run_time"] = standardize(
                sample["nvml_metrics"]["median_run_time"],
                runtime_mean,
                runtime_std,
            )
            for metric_name, value in sample["ncu_metrics"].items():
                sample["ncu_metrics"][metric_name] = standardize(
                    value,
                    ncu_metrics_means_stds[metric_name]["mean"],
                    ncu_metrics_means_stds[metric_name]["std"],
                )


if __name__ == "__main__":
    import json

    with open("training_data.json", "r") as f:
        data = json.load(f)

        dataset = CustomDataset(samples=data["training_data"], ptx=data["ptxs"])

    item = dataset.__getitem__(0)
    print(item)
