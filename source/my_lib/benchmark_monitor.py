import os
import sqlite3
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from .gpu import GPU, GPUQueries
from .utils import are_there_other_users

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
        nvml_n_runs: str,
        nvml_sampling_frequency: int,
        nsys_path: str,
        nsys_set: str,
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

        #################### NSYS config ####################

        self.__nsys_path = nsys_path
        """ The path of the NSYS profiler """

        self.__nsys_set = nsys_set
        """ The name of the set of metrics to collect using NSYS. """

    ################################### Public methods ####################################

    def run_nvml(
        self, benchmark_path: str, benchmark_args: list
    ) -> tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES, bool]:
        """
        Runs the benchmark and monitors it using nvml

        Parameters
        ----------
        benchmark_path: str
            The benchmark to run
        benchmarks_args: list
            The arguments that should be supplied to the benchmark

        Returns
        -------
        tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES, bool]
            - NVML_RESULTS_SUMMARY, RUN_SAMPLES -> Check docstring of __process_samples
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
                command = [f"./{benchmark_path}"]
                if benchmark_args is not None:
                    command += benchmark_args

                self.__wait_to_cooldown()
                sample_event.set()
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                sample_event.clear()

                # Wait for the other thread to append the samples and then collect them
                sampler_results_ready_event.wait()
                results.append(sampler_return_values[-1])

            ########## Post processing ##########

            terminate_event.set()

            summary_results, nvml_samples = self.__process_nvml_samples(
                all_run_samples=results
            )

            # Wait for thread to put the result
            users_results_ready_event.wait()

            return summary_results, nvml_samples, did_other_users_login[-1]
        finally:
            ########## Cleanup ##########
            terminate_event.set()
            sampler_thread.join()
            users_thread.join()

    def run_nsys(self, benchmark_path: str, benchmarks_args: list) -> tuple[dict, bool]:
        """
        Runs the benchmark and performs the profilling using nsys

        Parameters
        ----------
        benchmark_path: str
            The benchmark to run
        benchmarks_args: list
            The arguments that should be supplied to the benchmark

        Returns
        -------
        tuple[dict, bool]
            - A dict with the metrics collected
            - A boolean representing whether or not another user logged in during the sampling
        """

        benchmark_filename = os.path.basename(benchmark_path)
        report_filename = os.path.splitext(benchmark_filename)[0] + ".sqlite"
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
                    True,  # Let the thread know that NSYS will be running
                ),
            )
            users_thread.start()

            ############################## Run NSYS ##############################

            # Docs: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options
            # Example command:
            # sudo nsys profile --force-overwrite=true --gpu-metrics-set=ad10x-gfxt --gpu-metrics-device=0 --export=sqlite
            # --output=report.sqlite gpupowermodel_v1.0_HPCA2018_microbenchmarks_DRAM_dram.out

            command = [
                "sudo",
                self.__nsys_path,
                "profile",
                "--force-overwrite=true",
                f"--gpu-metrics-set={self.__nsys_set}",
                "--gpu-metrics-device=0",
                "--export=sqlite",
                f"--output={report_path}",
                benchmark_path,
            ]

            if benchmarks_args is not None:
                command += benchmarks_args

            self.__wait_to_cooldown()
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            terminate_event.set()  # NSYS already stopped collecting the metrics

            ############################## Collect results ##############################

            collected_metrics = self.__aggregate_nsys_metrics(report_path)

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

    def __aggregate_nsys_metrics(self, db_file: str) -> dict[str, float]:
        """
        Aggregate GPU metrics from the NSYS database

        Database schema: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?highlight=sqlite#sqlite-schema-reference

        Parameters
        ----------
        report_path: str
            The path where to find the NSYS report sqlite

        Returns
        -------
        dict[str, float]
            The dictionary whose keys are the metric's names and values are the metrics itself
        """

        conn = sqlite3.connect(db_file)

        try:

            # Query the table containing the gpu metric names and ids
            # selecting only the ones that are throughput metrics as they are relative and not absolute
            query_target_info = """
            SELECT sourceId, metricId, metricName
            FROM TARGET_INFO_GPU_METRICS
            WHERE metricName LIKE '%[Throughput %]' OR metricName LIKE '%[Ratio %]'
            """
            df_target_info = pd.read_sql(query_target_info, conn)

            metric_ids = df_target_info["metricId"].unique()

            # Check if all "sourceId" values are equal (i.e., same GPU)
            if df_target_info["sourceId"].nunique() != 1:
                raise ValueError("Not all sourceId values are equal.")
            source_id = df_target_info["sourceId"].iloc[0]

            # Query the gpu metrics (from the selected ones)
            # and that are from the same GPU (excluding the initial 'EndTimestamp' one)
            query_gpu_metrics = f"""
            SELECT metricId, value
            FROM GPU_METRICS
            WHERE metricId IN ({','.join(map(str, metric_ids))}) AND typeId = {source_id}
            """
            df_gpu_metrics = pd.read_sql(query_gpu_metrics, conn)

            # Compute the average for each metricId (across all timestamps)
            grouped_means = (
                df_gpu_metrics.groupby("metricId")["value"].mean().reset_index()
            )

            # Add metric names
            df_merged = pd.merge(
                grouped_means, df_target_info[["metricId", "metricName"]], on="metricId"
            )

            # Ensure consistent ordering
            df_merged.sort_values("metricId", inplace=True)

            avg_metrics = dict(zip(df_merged["metricName"], df_merged["value"]))

            return avg_metrics

        finally:
            conn.close()

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
        running_nsys: bool = False,
    ) -> None:
        """Thread that monitors whether or not other users used this machine (appends a boolean to `return_value`)"""

        did_other_users_login = False

        sleep_time = 30  # s (should be multiple of 1)

        while not terminate_event.is_set():

            if are_there_other_users(running_nsys=running_nsys):
                did_other_users_login = True
                break  # Thread can already exit since other users logged in during the benchmarking

            # Equivalent to `time.sleep(sleep_time)` but more responsive to the terminate event
            for _ in range(sleep_time):
                if terminate_event.is_set():
                    break
                time.sleep(1)

        return_value.append(did_other_users_login)
        results_ready_event.set()

    ################################### Others ####################################

    def __wait_to_cooldown(self):
        """Enters a loop that waits for the gpu to cooldown"""

        max_temp = 50  # ºC
        max_util = 5  # %

        i = 0
        while True:
            if (
                self.__gpu.query(GPUQueries.TEMPERATURE) < max_temp
                and self.__gpu.query(GPUQueries.GPU_UTILIZATION) < max_util
            ):
                break
            else:
                time.sleep(1)
                i += 1

            if i > 5:
                print(f"Have been waiting for the GPU to cooldown for {i} s...")
