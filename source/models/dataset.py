from typing import List

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset that expects data format exported by the benchmarks_to_training_data.py tool"""

    def __init__(self, samples: List, all_ptx: dict):

        self.all_ptx: dict[str : dict[str:List]] = self.__format_ptx(all_ptx)
        self.data: List = self.__convert_to_tensors(samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        benchmark_name = sample["benchmark_name"]

        split_ptx = self.all_ptx[benchmark_name]

        # Hardware and performance metrics
        core_freq = sample["graphics_frequency"]
        mem_freq = sample["memory_frequency"]
        cupti_metrics = torch.tensor(
            list(sample["benchmark_metrics"]["cupti_metrics"].values()),
            dtype=torch.float32,
        )  # Remove metrics names from the dictionary

        # Targets
        power_gold = sample["benchmark_metrics"]["nvml_metrics"]["POWER"]
        runtime_gold = sample["benchmark_metrics"]["runtime"]

        return {
            "benchmark_name": benchmark_name,
            "split_ptx": split_ptx,
            "graphics_frequency": core_freq,
            "memory_frequency": mem_freq,
            "cupti_metrics": cupti_metrics,
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

    def __format_ptx(self, all_ptx):
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

        for benchmark_names, kernels in all_ptx.items():
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


if __name__ == "__main__":
    import json

    with open("training_data.json", "r") as f:
        data = json.load(f)

        dataset = CustomDataset(samples=data["training_data"], all_ptx=data["ptxs"])

    item = dataset.__getitem__(0)
    print(item)
