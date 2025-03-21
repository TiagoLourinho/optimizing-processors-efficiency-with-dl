import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Any

import numpy as np

from .gpu import GPU, GPUQueries
from .utils import are_there_other_users

MatplotlibFigure = Any
""" 
Placeholder for the matplotlib figure class (not direclty imported because of C++ library incompatibilities)

Check: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
"""

RUN_SAMPLES = dict[str, list[float]]
""" 
The list of samples collected during a run per each metric type

Example: key GRAPHICS_CLOCK -> [810, 810, ... , 1995]
 """

NVML_RESULTS_SUMMARY = dict[str, float]
""" 
The dictionary containing the median value (across all runs) of the metrics average (within a run). 

Example: key average_GRAPHICS_CLOCK -> 810 
"""


class BenchmarkMonitor:
    """Monitors and samples GPU metrics while executing a CUDA benchmark"""

    ########## Others ##########

    METRICS = [
        # GPUQueries.GRAPHICS_CLOCK,
        # GPUQueries.MEMORY_CLOCK,
        # GPUQueries.TEMPERATURE,
        GPUQueries.POWER,
        # GPUQueries.GPU_UTILIZATION,
    ]
    """ The NVML metrics to collect from the GPU """

    ######################################## Dunder methods ########################################

    def __init__(
        self,
        gpu: GPU,
        nvcc_path: str,
        nvml_n_runs: str,
        nvml_sampling_frequency: int,
        ncu_path: str,
        ncu_sections_folder: str,
        ncu_python_report_folder: str,
        ncu_set: str,
    ) -> None:

        self.__gpu = gpu
        """ The GPU running the benchmark """

        #################### NVML config ####################

        self.__nvml_n_runs = nvml_n_runs
        """ The number of times to run the benchmark (to calculate the median results) """

        self.__nvml_sampling_frequency = min(
            nvml_sampling_frequency, 100
        )  # Limit NVML sampling frequency to 100 Hz
        """ The sampling frequency [Hz] """

        #################### NCU config ####################

        self.__ncu_path = ncu_path
        """ The path of the NCU profiler """

        self.__ncu_sections_folder = ncu_sections_folder
        """ The path of the folder to search for NCU sections. If None, then the default path is used. """

        self.__ncu_set = ncu_set
        """ The name of the set of metrics to collect using NCU. """

        sys.path.append(ncu_python_report_folder)
        self.__ncu_report = __import__("ncu_report")
        """ The NCU python report interface """

    ################################### Public methods ####################################

    def run_nvml(
        self, benchmark_path: str
    ) -> tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES, MatplotlibFigure, bool]:
        """
        Runs the benchmark and monitors it using nvml

        Parameters
        ----------
        benchmark_path: str
            The benchmark to run

        Returns
        -------
        tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES, MatplotlibFigure, bool]
            - NVML_RESULTS_SUMMARY, RUN_SAMPLES -> Check docstring of __process_samples
            - A figure with the execution plots
            - A boolean representing whether or not another user logged in during the sampling
        """

        try:
            ########## Sampler thread management ##########

            sampler_return_values = (
                []
            )  # Samples collected by the thread will be appended here
            sampler_results_ready_event = (
                threading.Event()
            )  # Signals when the samples were appended to the return list

            sample_event = (
                threading.Event()
            )  # Controls when the thread should be sampling
            terminate_event = (
                threading.Event()
            )  # Controls when the thread should terminate

            sampler_thread = threading.Thread(
                target=self.__thread_sample_gpu,
                args=(
                    sampler_return_values,
                    sampler_results_ready_event,
                    sample_event,
                    terminate_event,
                ),
            )
            sampler_thread.start()

            ########## Users monitoring thread management ##########

            # Thread will append a boolean here representing
            # whether or not another user logged in during benchmarking
            did_other_users_login = []
            users_results_ready_event = (
                threading.Event()
            )  # Signals when the return boolean was appended to the return list so it can be collected

            users_thread = threading.Thread(
                target=self.__thread_monitor_users,
                args=(
                    did_other_users_login,
                    users_results_ready_event,
                    terminate_event,
                ),
            )
            users_thread.start()

            ########## Run benchmark and collect results ##########
            results = []

            for _ in range(self.__nvml_n_runs):

                # Clean events and return list before running
                sample_event.clear()
                sampler_results_ready_event.clear()
                sampler_return_values.clear()

                # Start sampling and run the application
                sample_event.set()
                subprocess.run(
                    [f"./{benchmark_path}"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                sample_event.clear()

                # Wait for the other thread to append the samples and then collect them
                sampler_results_ready_event.wait()
                results.append(sampler_return_values[-1])

                time.sleep(self.__gpu.sleep_time)

            ########## Post processing ##########

            terminate_event.set()

            summary_results, nvml_samples = self.__process_nvml_samples(
                all_run_samples=results
            )
            figure = self.__create_nvml_plots(run_samples=nvml_samples)

            # Wait for thread to put the result
            users_results_ready_event.wait()

            return summary_results, nvml_samples, figure, did_other_users_login[-1]
        finally:
            ########## Cleanup ##########
            terminate_event.set()
            sampler_thread.join()
            users_thread.join()

    def run_ncu(self, benchmark_path: str) -> tuple[dict, bool]:
        """
        Runs the benchmark and performs the benchmork using ncu

        Parameters
        ----------
        benchmark_path: str
            The benchmark to run

        Returns
        -------
        tuple[dict, bool]
            - A dict with the metrics collected
            - A boolean representing whether or not another user logged in during the sampling
        """

        benchmark_filename = os.path.basename(benchmark_path)
        report_filename = os.path.splitext(benchmark_filename)[0] + ".ncu-rep"
        report_dir = os.path.join("bin", "reports")
        report_path = os.path.join(report_dir, report_filename)
        os.makedirs(report_dir, exist_ok=True)

        try:

            ############################## Users monitoring thread management ##############################

            # Thread will append a boolean here representing
            # whether or not another user logged in during benchmarking
            did_other_users_login = []
            users_results_ready_event = (
                threading.Event()
            )  # Signals when the return boolean was appended to the return list so it can be collected
            terminate_event = (
                threading.Event()
            )  # Controls when the thread should terminate

            users_thread = threading.Thread(
                target=self.__thread_monitor_users,
                args=(
                    did_other_users_login,
                    users_results_ready_event,
                    terminate_event,
                    True,  # Let the thread know that NCU will be running
                ),
            )
            users_thread.start()

            ############################## Run NCU ##############################

            # Docs: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options
            # Example command:
            # sudo /usr/local/cuda-12.4/bin/ncu -f --clock-control none -o bin/tiny.ncu-rep
            # --section-folder /opt/nvidia/nsight-compute/2024.1.1/sections --set basic bin/tiny.out

            command = [
                "sudo",
                self.__ncu_path,
                "-f",
                "--clock-control",  # clock-control=none allows NVML to externally manage the frequencies
                "none",
                "-o",
                report_path,
            ]

            if self.__ncu_sections_folder is not None:
                command += ["--section-folder", self.__ncu_sections_folder]

            command += [
                "--set",
                self.__ncu_set,
                benchmark_path,
            ]

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            terminate_event.set()  # NCU already stopped collecting the metrics

            ############################## Collect results ##############################

            collected_metrics = self.__aggregate_ncu_metrics(report_path)

            # Wait for thread to put the result before collecting it
            users_results_ready_event.wait()

            return collected_metrics, did_other_users_login[-1]
        finally:
            ########## Cleanup ##########
            terminate_event.set()

            users_thread.join()

    ################################### Data post processing ####################################

    def __process_nvml_samples(
        self, all_run_samples: list[RUN_SAMPLES]
    ) -> tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES]:
        """
        Given all the samples collected for each NVML run,
        calculates the average value per each metric within each run and
        then calculates the median value across runs,
        yielding the "results summary"

        Parameters
        ----------
        all_run_samples: list[SAMPLES_PER_TYPE]
            The list of length 'N_runs' containing the output of __thread_sample_gpu method per each run

        Returns
        -------
        tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES]
            - The dictionary with the summary of the main metrics collected
                Example:
                    {
                        average_GRAPHICS_CLOCK: 1600,
                        average_TEMPERATURE: 67,
                        ...
                        median_run_time: 23
                    }
            - The NVML samples of the metrics collected that can be plotted
            (corresponding to the run with the runtime closest to the median)
                Example:
                    Check docstring of __thread_sample_gpu method
        """
        n_runs = len(all_run_samples)
        n_metrics = len(self.METRICS)

        # Compute the averages of each metric per run
        averages_per_run = np.full((n_runs, n_metrics), np.nan)
        run_times_per_run = np.full(n_runs, np.nan)

        for run_index in range(n_runs):
            for metric_index, metric in enumerate(self.METRICS):
                averages_per_run[run_index, metric_index] = np.average(
                    all_run_samples[run_index][metric.name]
                )

            # Collect the run time (last sample time)
            run_times_per_run[run_index] = all_run_samples[run_index]["sample_time"][-1]

        # Compute the median value per metric considering all the runs
        median_values = {}
        for metric_index, metric in enumerate(self.METRICS):
            median_values[f"average_{metric.name}"] = float(
                np.median(averages_per_run[:, metric_index])
            )
        median_values["median_run_time"] = float(np.median(run_times_per_run))

        # Get the index of the run that had the closest run time to the median
        index = np.argmin(
            np.absolute(run_times_per_run - median_values["median_run_time"])
        )

        return median_values, all_run_samples[index]

    def __create_nvml_plots(self, run_samples: RUN_SAMPLES) -> MatplotlibFigure:
        """Creates the matplotlib figure with the metrics samples for a run"""

        # Import locally as NCU and matplotlib use different C++ binaries
        # and that would result in the following error:
        #
        # terminate called after throwing an instance of 'std::bad_cast'
        # what():  std::bad_cast
        import matplotlib.pyplot as plt

        n_metrics = len(self.METRICS)
        n_columns = 2
        n_rows = int(np.ceil(n_metrics / n_columns))

        fig, axs = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 3 * n_rows))

        # Plot the results
        flat_axs = axs.flat
        for metric_index, metric in enumerate(self.METRICS):

            # Line
            flat_axs[metric_index].plot(
                run_samples["sample_time"],
                run_samples[metric.name],
                color="blue",
                zorder=0,
            )

            # Sample points
            flat_axs[metric_index].scatter(
                run_samples["sample_time"],
                run_samples[metric.name],
                marker="o",
                s=1,
                color="red",
                zorder=1,
            )

            flat_axs[metric_index].set_xlabel("Run time [s]")
            flat_axs[metric_index].set_ylabel(
                metric.value
            )  # The value of the enum contains the name and unit

            flat_axs[metric_index].grid(True)

        # Remove unused plots
        if n_metrics < n_rows * n_columns:
            for i in range(n_metrics, n_rows * n_columns):
                fig.delaxes(flat_axs[i])

        fig.tight_layout()
        fig.subplots_adjust(top=0.925)  # Spacing for title

        return fig

    def __aggregate_ncu_metrics(self, report_path: str) -> dict[str, float]:
        """
        Aggregates the NCU metrics (considering multiple ranges and actions) and returns the dictionary with the metrics

        Parameters
        ----------
        report_path: str
            The path where to find the NCU report object

        Returns
        -------
        dict[str, float]
            The dictionary whose keys are the metric's names and values are the metrics itself
        """

        # Examples and docs of using NCU python report interface:
        # https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#basic-usage

        context = self.__ncu_report.load_report(report_path)

        # Identifies launch or device attributes so they can be removed
        other_type = getattr(self.__ncu_report, "IMetric").MetricType_OTHER

        N_total_kernels = sum(
            [
                context.range_by_idx(range_i).num_actions()
                for range_i in range(context.num_ranges())
            ]
        )

        # Loop through all ranges and actions to aggregate by averaging
        collected_metrics = defaultdict(float)
        for ncu_range_i in range(context.num_ranges()):
            ncu_range = context.range_by_idx(ncu_range_i)

            for action_j in range(ncu_range.num_actions()):
                action = ncu_range.action_by_idx(action_j)

                for metric_name in action.metric_names():
                    metric = action[metric_name]
                    metric_type = metric.metric_type()
                    metric_value = metric.value()

                    if metric_type != other_type:
                        # Average the partials already
                        collected_metrics[metric_name] += metric_value / N_total_kernels

        return collected_metrics

    ################################### Threaded methods ####################################

    def __thread_sample_gpu(
        self,
        return_values: list,
        results_ready_event: threading.Event,
        sample_event: threading.Event,
        terminate_event: threading.Event,
    ) -> None:
        """
        Method that should be called in a auxiliar thread to sample the GPU metrics

        Parameters
        ----------
        return_values: list
            A list where the return values (GPU metrics) of this thread will be appended
        results_ready_event: threading.Event
            An event to signal the main thread when the samples were appended to `return_values` meaning they are ready to be collected
        sample_event: threading.Event
            An event controlling whether or not this thread should be sampling the GPU
        terminate_event: threading.Event
            An event controlling when this thread should terminate


        Returns
        -------
        SAMPLES_PER_TYPE (appended to `return_values`)
            Example:
                {
                    GRAPHICS_CLOCK: [1500, 1550, 1600, 1575, 1620],
                    TEMPERATURE: [65, 67, 68, 66, 70],
                    ...
                    sample_time: [ 0, 0.1, 0.2, 0.3, 0.4] # Seconds since the start of the sampling
                }
        """

        period = 1 / self.__nvml_sampling_frequency

        while not terminate_event.is_set():

            # Initialize empty samples lists and wait to start
            samples = {metric.name: [] for metric in self.METRICS}
            samples["sample_time"] = []  # Seconds since the start of the sampling
            sample_time = 0

            flag = sample_event.wait(timeout=0.5)

            if flag:
                while sample_event.is_set() and not terminate_event.is_set():

                    # Collect samples
                    for metric in self.METRICS:
                        samples[metric.name].append(self.__gpu.query(query_type=metric))

                    samples["sample_time"].append(sample_time)
                    time.sleep(period)
                    sample_time += period

                # "Return" the collected samples
                return_values.append(samples)
                results_ready_event.set()

    def __thread_monitor_users(
        self,
        return_value: list[bool],
        results_ready_event: threading.Event,
        terminate_event: threading.Event,
        running_ncu: bool = False,
    ) -> None:
        """Thread that monitors whether or not other users used this machine (appends a boolean to `return_value`)"""

        did_other_users_login = False

        sleep_time = 30  # s (should be multiple of 1)

        while not terminate_event.is_set():

            if are_there_other_users(running_ncu=running_ncu):
                did_other_users_login = True
                break  # Thread can already exit since other users logged in during the benchmarking

            # Equivalent to `time.sleep(sleep_time)` but more responsive to the terminate event
            for _ in range(sleep_time):
                if terminate_event.is_set():
                    break
                time.sleep(1)

        return_value.append(did_other_users_login)
        results_ready_event.set()
