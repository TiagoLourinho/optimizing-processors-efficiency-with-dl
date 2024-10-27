import os
import subprocess
import sys
import threading
import time

import numpy as np
from tqdm import tqdm

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

Example: key median_GRAPHICS_CLOCK -> 810 
"""


class BenchmarkMonitor:
    """
    Monitors and samples GPU metrics while executing a CUDA benchmark

    Assumes that the benchmark writes the necessary events to stdout in real time (flushing the buffer), namely like:

    `std::cout << "EVENT:START_ROI" << std::endl;`

    and then

    `std::cout << "EVENT:END_ROI" << std::endl;`

    """

    ########## Events ##########

    START_ROI_EVENT = "EVENT:START_ROI"
    """ Event defining the start of the Region Of Interest """

    END_ROI_EVENT = "EVENT:END_ROI"
    """ Event defining the end of the Region Of Interest """

    ########## Others ##########

    METRICS = [
        GPUQueries.GRAPHICS_CLOCK,
        GPUQueries.MEMORY_CLOCK,
        GPUQueries.TEMPERATURE,
        GPUQueries.POWER,
        GPUQueries.GPU_UTILIZATION,
    ]
    """ The metrics to collect from the GPU """

    bin_folder = "bin"
    """ The name of the bin folder """

    ######################################## Dunder methods ########################################

    def __init__(
        self,
        benchmark: str,
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

        self.__benchmark = self.__compile(cuda_file=benchmark, nvcc_path=nvcc_path)
        """ The path of the CUDA binary to monitor """

        #################### NVML config ####################

        self.__nvml_n_runs = nvml_n_runs
        """ The number of times to run the benchmark (to calculate the median results) """

        self.__nvml_sampling_frequency = min(
            nvml_sampling_frequency, 50
        )  # Limit NVML sampling frequency to 50 Hz
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
        self,
    ) -> tuple[
        NVML_RESULTS_SUMMARY, RUN_SAMPLES, any, bool
    ]:  # any -> matplotlib.figure.Figure (see __create_plots for more info)
        """
        Runs the benchmark and monitors it using nvml

        Returns
        -------
        tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES, matplotlib.figure.Figure, bool]
            - NVML_RESULTS_SUMMARY, RUN_SAMPLES -> Check docstring of __process_samples
            - A figure with the execution plots
            - A boolean representing whether or not another user logged in during the sampling
        """

        print("Collecting kernel metrics with NVML...")

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

            for _ in tqdm(range(self.__nvml_n_runs)):

                # Clean events and return list before running
                sample_event.clear()
                sampler_results_ready_event.clear()
                sampler_return_values.clear()

                # Check stdout of the benchmark and sample inside the Region Of Interest
                for stdout_line in self.__run_command_generator(
                    args=[self.__benchmark]
                ):

                    if (
                        self.START_ROI_EVENT in stdout_line
                        and not sample_event.is_set()
                    ):
                        sample_event.set()
                    elif self.END_ROI_EVENT in stdout_line and sample_event.is_set():

                        sample_event.clear()

                        # Wait for the other thread to append the samples and then collect them
                        sampler_results_ready_event.wait()
                        results.append(sampler_return_values[-1])

            ########## Post processing ##########

            terminate_event.set()

            summary_results, nvml_samples = self.__process_samples(
                all_run_samples=results
            )
            figure = self.__create_plots(run_samples=nvml_samples)

            # Wait for thread to put the result
            users_results_ready_event.wait()

            return summary_results, nvml_samples, figure, did_other_users_login[-1]
        finally:
            ########## Cleanup ##########
            terminate_event.set()
            sampler_thread.join()
            users_thread.join()

    def run_ncu(self) -> tuple[dict, bool]:
        """
        Runs the benchmark and performs the benchmork using ncu

        Returns
        -------
        tuple[dict, bool]
            - A dict with the metrics collected
            - A boolean representing whether or not another user logged in during the sampling
        """

        print("Collecting kernel metrics with NCU...")

        report_path = os.path.join(
            self.bin_folder,
            os.path.basename(self.__benchmark).replace(".out", ".ncu-rep"),
        )

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
                ),
            )
            users_thread.start()

            ############################## Run NCU ##############################

            # Docs: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options
            # Example command:
            # sudo /usr/local/cuda-12.4/bin/ncu -f --clock-control none -o bin/tiny.ncu-rep --section-folder /opt/nvidia/nsight-compute/2024.1.1/sections --set basic bin/tiny.out

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

            command += ["--set", self.__ncu_set, self.__benchmark]

            # Use list to fully run the generator
            list(self.__run_command_generator(args=command))

            terminate_event.set()  # NCU already stopped collecting the metrics

            ############################## Collect results ##############################

            # Examples and docs:
            # https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#basic-usage

            context = self.__ncu_report.load_report(report_path)

            if context.num_ranges() != 1:
                raise ValueError("Tool is only expecting 1 range to exist.")

            range = context.range_by_idx(0)

            if range.num_actions() != 1:
                raise ValueError("Tool is only expecting 1 kernel to exist.")

            action = range.action_by_idx(0)

            # Identifies launch or device attributes so they can be removed
            other_type = getattr(self.__ncu_report, "IMetric").MetricType_OTHER

            collected_metrics = {
                metric_name: action[metric_name].value()
                for metric_name in action.metric_names()
                if action[metric_name].metric_type()
                != other_type  # Filter our non hardware metrics
            }

            # Wait for thread to put the result before collecting it
            users_results_ready_event.wait()

            return collected_metrics, did_other_users_login[-1]
        finally:
            ########## Cleanup ##########
            terminate_event.set()

            users_thread.join()

    ################################### Utils ####################################

    def __compile(self, cuda_file: str, nvcc_path: str) -> str:
        """Compiles the CUDA program"""

        print("Compiling benchmark...")

        if not os.path.exists(self.bin_folder):
            os.makedirs(self.bin_folder)

        output_path = os.path.join(
            self.bin_folder, os.path.basename(cuda_file).replace(".cu", ".out")
        )

        nvcc_command = [
            nvcc_path,
            cuda_file,
            "-o",
            output_path,
        ]

        # Use list to fully run the generator command
        list(self.__run_command_generator(args=nvcc_command))

        return output_path

    def __run_command_generator(self, args: list[str]):
        """Runs a command defined by `args` and acts as a generator for stdout"""

        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        while True:
            output = process.stdout.readline().strip()
            return_code = process.poll()

            # process.poll returns None while the process is still running
            if return_code is not None:
                break
            elif output:
                yield output

        if return_code != 0:
            raise Exception(
                f"There was an error running '{' '.join(args)}':\n\n{process.stderr.read().strip()}"
            )

    ################################### Data post processing ####################################

    def __process_samples(
        self, all_run_samples: list[RUN_SAMPLES]
    ) -> tuple[NVML_RESULTS_SUMMARY, RUN_SAMPLES]:
        """
        Given all the samples collected for each run,
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
                        median_GRAPHICS_CLOCK: 1600,
                        median_TEMPERATURE: 67,
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
            median_values[f"median_{metric.name}"] = float(
                np.median(averages_per_run[:, metric_index])
            )
        median_values["median_run_time"] = float(np.median(run_times_per_run))

        # Get the index of the run that had the closest run time to the median
        index = np.argmin(
            np.absolute(run_times_per_run - median_values["median_run_time"])
        )

        return median_values, all_run_samples[index]

    def __create_plots(
        self, run_samples: RUN_SAMPLES
    ) -> any:  # any -> matplotlib.figure.Figure (see below for more info)
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
        fig.suptitle(
            f"'{os.path.basename(self.__benchmark).replace('.out','.cu')}' execution plots",
            fontsize=16,
        )

        return fig

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
    ) -> None:
        """Thread that monitors whether or not other users used this machine (appends a boolean to `return_value`)"""

        did_other_users_login = False

        sleep_time = 30  # s (should be multiple of 1)

        while not terminate_event.is_set():

            if are_there_other_users():
                did_other_users_login = True
                break  # Thread can already exit since other users logged in during the benchmarking

            # Equivalent to `time.sleep(sleep_time)` but more responsive to the terminate event
            for _ in range(sleep_time):
                if terminate_event.is_set():
                    break
                time.sleep(1)

        return_value.append(did_other_users_login)
        results_ready_event.set()
