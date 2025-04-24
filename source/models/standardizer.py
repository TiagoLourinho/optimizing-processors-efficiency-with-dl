import torch


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
        power_all_values: list[float] = []
        runtime_all_values: list[float] = []
        ncu_metrics_all_values: dict[str : list[float]] = (
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
            power_all_values.append(sample["nvml_metrics"]["average_POWER"])
            runtime_all_values.append(sample["nvml_metrics"]["median_run_time"])

            for metric_name, value in sample["ncu_metrics"].items():
                if metric_name not in ncu_metrics_all_values:
                    ncu_metrics_all_values[metric_name] = []
                ncu_metrics_all_values[metric_name].append(value)

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
        power_all_values = torch.tensor(power_all_values, dtype=torch.float32)
        runtime_all_values = torch.tensor(runtime_all_values, dtype=torch.float32)
        ncu_metrics_all_values = {
            metric_name: torch.tensor(values, dtype=torch.float32)
            for metric_name, values in ncu_metrics_all_values.items()
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
        self.power_mean, self.power_std = (
            power_all_values.mean(),
            power_all_values.std(),
        )
        self.runtime_mean, self.runtime_std = (
            runtime_all_values.mean(),
            runtime_all_values.std(),
        )
        self.ncu_metrics_means_stds = {
            metric_name: {"mean": values.mean(), "std": values.std()}
            for metric_name, values in ncu_metrics_all_values.items()
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
            sample["nvml_metrics"]["average_POWER"] = self.__standardize(
                sample["nvml_metrics"]["average_POWER"],
                self.power_mean,
                self.power_std,
            )
            sample["nvml_metrics"]["median_run_time"] = self.__standardize(
                sample["nvml_metrics"]["median_run_time"],
                self.runtime_mean,
                self.runtime_std,
            )
            for metric_name, value in sample["ncu_metrics"].items():
                sample["ncu_metrics"][metric_name] = self.__standardize(
                    value,
                    self.ncu_metrics_means_stds[metric_name]["mean"],
                    self.ncu_metrics_means_stds[metric_name]["std"],
                )

    def inv_transform_targets(
        self,
        power: torch.Tensor,
        runtime: torch.Tensor,
    ) -> tuple[float, float]:
        """Inverse transform the power and runtime values to their original scale."""

        if not self.fitted:
            raise ValueError("Standardizer has not been fitted yet.")

        power = float(power.cpu().item())
        runtime = float(runtime.cpu().item())

        return (
            self.__destandardize(power, self.power_mean, self.power_std),
            self.__destandardize(runtime, self.runtime_mean, self.runtime_std),
        )

    def __standardize(self, value: float, mean: float, std: float) -> float:
        """Standaridzation formula"""

        # Avoid division by zero in some ncu metrics that are constant
        if std == 0.0:
            return 0.0
        else:
            return float((value - mean) / std)

    def __destandardize(self, value: float, mean: float, std: float) -> float:
        """Destandarization formula"""

        return float((value * std) + mean)
