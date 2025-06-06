import torch
import pickle


class Standardizer:
    """
    Standarizes the data by removing the mean and scaling to unit variance.

    Easier to understand while looking at the data format in training_data.json

    The steps are:
        1. Collect all values from the ptx numerical part and the samples (fit method)
        2. Calculate the mean and std for each feature (fit method)
        3. Standardize the values using the formula: (value - mean) / std (transform_ptx/samples method)
        4. Replace the original values with the standardized ones (transform_ptx/samples method)
    """

    def __init__(self, ptx_n_numerical_features: int):
        self.fitted = False
        self.ptx_n_numerical_features = ptx_n_numerical_features

    def save(self, path: str) -> None:
        """Saves the fitted Standardizer instance to a file using pickle."""

        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before saving.")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Standardizer":
        """Loads a Standardizer instance from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def fit(
        self,
        train_samples: list[dict],
        all_ptx: dict[str:dict],
    ) -> None:
        """
        Finds the means and stds of the input data.

        All samples will be used to calculate the means and stds of the features, however,
        only the ptx that appear on the samples will be used to calculate the means and stds of the ptx numerical part.
        """

        ######### Initialization #########

        mem_freq_all_values: list[int] = []
        core_freq_all_values: list[int] = []
        runtime_all_values: list[float] = []
        nvml_metrics_all_values: dict[str : list[float]] = (
            {}
        )  # The list for each metric will be later initialized
        cupti_metrics_all_values: dict[str : list[float]] = (
            {}
        )  # The list for each metric will be later initialized
        targets_all_values: dict[str : list[float]] = (
            {}
        )  # The list for each metric will be later initialized
        ptx_numerical_part_all_values: list[list[int]] = [
            [] for _ in range(self.ptx_n_numerical_features)
        ]

        ######### All values collection #########
        seen_benchmarks_ptx = set()

        for sample in train_samples:
            seen_benchmarks_ptx.add(sample["benchmark_name"])
            mem_freq_all_values.append(sample["memory_frequency"])
            core_freq_all_values.append(sample["graphics_frequency"])
            runtime_all_values.append(sample["benchmark_metrics"]["runtime"])

            for metric_name, value in sample["benchmark_metrics"][
                "nvml_metrics"
            ].items():
                if metric_name not in nvml_metrics_all_values:
                    nvml_metrics_all_values[metric_name] = []
                nvml_metrics_all_values[metric_name].append(value)

            for metric_name, value in sample["benchmark_metrics"][
                "cupti_metrics"
            ].items():
                if metric_name not in cupti_metrics_all_values:
                    cupti_metrics_all_values[metric_name] = []
                cupti_metrics_all_values[metric_name].append(value)

            for target_name, value in sample["targets"].items():
                if target_name not in targets_all_values:
                    targets_all_values[target_name] = []
                targets_all_values[target_name].append(value)

        for benchmark_name, benchmark_kernels in all_ptx.items():

            # Only consider the ptx that appear on the training samples
            if benchmark_name not in seen_benchmarks_ptx:
                continue

            for kernel_instructions in benchmark_kernels.values():
                for instruction in kernel_instructions:
                    for dimension in range(self.ptx_n_numerical_features):
                        ptx_numerical_part_all_values[dimension].append(
                            instruction["numerical"][dimension]
                        )

        ######### Convertion to tensors #########

        mem_freq_all_values = torch.tensor(mem_freq_all_values, dtype=torch.float32)
        core_freq_all_values = torch.tensor(core_freq_all_values, dtype=torch.float32)
        runtime_all_values = torch.tensor(runtime_all_values, dtype=torch.float32)
        nvml_metrics_all_values = {
            metric_name: torch.tensor(values, dtype=torch.float32)
            for metric_name, values in nvml_metrics_all_values.items()
        }
        cupti_metrics_all_values = {
            metric_name: torch.tensor(values, dtype=torch.float32)
            for metric_name, values in cupti_metrics_all_values.items()
        }
        targets_all_values = {
            target_name: torch.tensor(values, dtype=torch.float32)
            for target_name, values in targets_all_values.items()
        }
        ptx_numerical_part_all_values = [
            torch.tensor(values, dtype=torch.float32)
            for values in ptx_numerical_part_all_values
        ]

        ######### Means and stds collections #########

        self.mem_freq_mean, self.mem_freq_std = (
            mem_freq_all_values.mean(),
            mem_freq_all_values.std(),
        )
        self.core_freq_mean, self.core_freq_std = (
            core_freq_all_values.mean(),
            core_freq_all_values.std(),
        )
        self.runtime_mean, self.runtime_std = (
            runtime_all_values.mean(),
            runtime_all_values.std(),
        )
        self.nvml_metrics_means_stds = {
            metric_name: {"mean": values.mean(), "std": values.std()}
            for metric_name, values in nvml_metrics_all_values.items()
        }
        self.cupti_metrics_means_stds = {
            metric_name: {"mean": values.mean(), "std": values.std()}
            for metric_name, values in cupti_metrics_all_values.items()
        }
        self.targets_means_stds = {
            target_name: {"mean": values.mean(), "std": values.std()}
            for target_name, values in targets_all_values.items()
        }
        self.ptx_numerical_part_means_stds = [
            {"mean": values.mean(), "std": values.std()}
            for values in ptx_numerical_part_all_values
        ]

        self.fitted = True

    def transform_ptx(self, all_ptx: dict) -> None:
        """Transforms all ptx using the fitted standardizer (that should have been only fitted on the training data)."""

        if not self.fitted:
            raise ValueError("Standardizer has not been fitted yet.")

        for benchmark_kernels in all_ptx.values():
            for kernel_instructions in benchmark_kernels.values():
                for instruction in kernel_instructions:
                    for dimension in range(self.ptx_n_numerical_features):
                        instruction["numerical"][dimension] = self.__standardize(
                            instruction["numerical"][dimension],
                            self.ptx_numerical_part_means_stds[dimension]["mean"],
                            self.ptx_numerical_part_means_stds[dimension]["std"],
                        )

    def transform_samples(self, samples: list) -> None:
        """Transforms the samples using the fitted standardizer."""

        if not self.fitted:
            raise ValueError("Standardizer has not been fitted yet.")

        for sample in samples:
            sample["memory_frequency"] = self.__standardize(
                sample["memory_frequency"],
                self.mem_freq_mean,
                self.mem_freq_std,
            )
            sample["graphics_frequency"] = self.__standardize(
                sample["graphics_frequency"],
                self.core_freq_mean,
                self.core_freq_std,
            )
            sample["benchmark_metrics"]["runtime"] = self.__standardize(
                sample["benchmark_metrics"]["runtime"],
                self.runtime_mean,
                self.runtime_std,
            )
            for metric_name, value in sample["benchmark_metrics"][
                "nvml_metrics"
            ].items():
                sample["benchmark_metrics"]["nvml_metrics"][metric_name] = (
                    self.__standardize(
                        value,
                        self.nvml_metrics_means_stds[metric_name]["mean"],
                        self.nvml_metrics_means_stds[metric_name]["std"],
                    )
                )
            for metric_name, value in sample["benchmark_metrics"][
                "cupti_metrics"
            ].items():
                sample["benchmark_metrics"]["cupti_metrics"][metric_name] = (
                    self.__standardize(
                        value,
                        self.cupti_metrics_means_stds[metric_name]["mean"],
                        self.cupti_metrics_means_stds[metric_name]["std"],
                    )
                )

            for target_name, value in sample["targets"].items():
                sample["targets"][target_name] = self.__standardize(
                    value,
                    self.targets_means_stds[target_name]["mean"],
                    self.targets_means_stds[target_name]["std"],
                )

    def inv_transform_targets(
        self,
        memory_scaling_factor: torch.Tensor,
        graphics_scaling_factor: torch.Tensor,
    ) -> tuple[float, float]:
        """Inverse transform the scaling factors values to their original scale."""

        if not self.fitted:
            raise ValueError("Standardizer has not been fitted yet.")

        memory_scaling_factor = float(memory_scaling_factor.cpu().item())
        graphics_scaling_factor = float(graphics_scaling_factor.cpu().item())

        # Use max to ensure non-negative scaling factors
        return (
            max(
                0,
                self.__destandardize(
                    memory_scaling_factor,
                    self.targets_means_stds["memory_scaling_factor"]["mean"],
                    self.targets_means_stds["memory_scaling_factor"]["std"],
                ),
            ),
            max(
                0,
                self.__destandardize(
                    graphics_scaling_factor,
                    self.targets_means_stds["graphics_scaling_factor"]["mean"],
                    self.targets_means_stds["graphics_scaling_factor"]["std"],
                ),
            ),
        )

    def __standardize(self, value: float, mean: float, std: float) -> float:
        """Standaridzation formula"""

        # Avoid division by zero in some cupti metrics that are constant
        if std == 0.0:
            return 0.0
        else:
            return float((value - mean) / std)

    def __destandardize(self, value: float, mean: float, std: float) -> float:
        """Destandarization formula"""

        return float((value * std) + mean)
