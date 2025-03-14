from typing import List

import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """Training dataset that expects data format exported by the benchmarks_to_training_data.py tool"""

    def __init__(self, samples: List, ptx: dict):
        self.ptxs: dict[str : dict[str:List]] = self.__convert_to_tensors(ptx)
        self.data: List = self.__convert_to_tensors(samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        benchmark_name = sample["benchmark_name"]

        ptx = self.ptxs[benchmark_name]

        # Hardware and performance metrics
        core_freq = sample["graphics_frequency"]
        mem_freq = sample["memory_frequency"]
        ncu_metrics = torch.tensor(
            list(sample["ncu_metrics"].values()), dtype=torch.float32
        )  # Remove metrics names from the dictionary

        # Targets
        avg_power = sample["nvml_metrics"]["average_POWER"]
        median_runtime = sample["nvml_metrics"]["median_run_time"]

        return {
            "ptx": ptx,
            "core_freq": core_freq,
            "mem_freq": mem_freq,
            "ncu_metrics": ncu_metrics,
            "avg_power": avg_power,
            "median_runtime": median_runtime,
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


if __name__ == "__main__":
    import json

    with open("training_data.json", "r") as f:
        data = json.load(f)

        dataset = TrainingDataset(data)

    item = dataset.__getitem__(1)
    print(item)
