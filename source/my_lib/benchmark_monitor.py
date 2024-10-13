import os
import subprocess
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .gpu import GPU, GPUQueries
from .utils import are_there_other_users


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

    METRICS = [
        GPUQueries.GRAPHICS_CLOCK,
        GPUQueries.MEMORY_CLOCK,
        GPUQueries.TEMPERATURE,
        GPUQueries.POWER,
        GPUQueries.GPU_UTILIZATION,
    ]
    """ The metrics to collect from the GPU """

    ######################################## Dunder methods ########################################

    def __init__(
        self,
        benchmark: str,
        gpu: GPU,
        nvcc_path: str,
        NVML_N_runs: str,
        NVML_sampling_frequency: int,
    ) -> None:

        self.__gpu = gpu
        """ The GPU running the benchmark """

        self.__benchmark = self.__compile(cuda_file=benchmark, nvcc_path=nvcc_path)
        """ The path of the CUDA binary to monitor """

        self.__NVML_N_runs = NVML_N_runs
        """ The number of times to run the benchmark (to calculate the median results) """

        self.__NVML_sampling_frequency = min(
            NVML_sampling_frequency, 50
        )  # Limit NVML sampling frequency to 50 Hz
        """ The sampling frequency [Hz] """

    ################################### Public methods ####################################

    def run_and_monitor(self) -> tuple[dict, dict, matplotlib.figure.Figure]:
        """
        Runs the benchmark and monitors it

        Returns
        -------
        tuple[dict, dict, matplotlib.figure.Figure]
            - dicts -> Check docstring of __process_samples + the summary one contains a `did_other_users_login` boolean key
            - A figure with the execution plots
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
            )  # Controls when the thread shoud terminate

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
            print("Running benchmark and collecting samples...")
            for _ in tqdm(range(self.__NVML_N_runs)):

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

            summary_results, timeline = self.__process_samples(samples=results)
            figure = self.__create_plots(timeline=timeline)

            # Wait for thread to put the result and then add it to the summary
            users_results_ready_event.wait()
            summary_results["did_other_users_login"] = did_other_users_login[-1]

            return summary_results, timeline, figure
        finally:
            ########## Cleanup ##########
            terminate_event.set()
            sampler_thread.join()
            users_thread.join()

    ################################### Utils ####################################

    def __compile(self, cuda_file: str, nvcc_path: str) -> str:
        """Compiles the CUDA program"""

        print("Compiling benchmark...")

        bin_folder = "bin"

        if not os.path.exists(bin_folder):
            os.makedirs(bin_folder)

        output_path = os.path.join(
            bin_folder, os.path.basename(cuda_file).replace(".cu", ".out")
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

    def __process_samples(self, samples: list[dict[str, list[float]]]):
        """
        Given the raw `samples` list calculates the median and returns the summary results and the timeline

        Parameters
        ----------
        samples: list[dict[str, list[float]]]
            The list of length 'N_runs' containing the output of __sample_gpu_thread method per each run

        Returns
        -------
        tuple[dict, dict]
            - The dictionary with the summary of the main metrics collected (the median was used across runs)
                Example:
                    {
                        median_GRAPHICS_CLOCK: 1600,
                        median_TEMPERATURE: 67,
                        ...
                        median_run_time: 23
                    }
            - The timeline of the metrics collected that can be plotted
                Example:
                    Check docstring of __sample_gpu_thread method
        """
        n_runs = len(samples)
        n_metrics = len(self.METRICS)

        # Compute the medians of each metric per run
        # (the median is used to reduce the impact of potencial spikes on the metrics caused by the sensors or etc)
        medians_per_run = np.full((n_runs, n_metrics), np.nan)
        run_times_per_run = np.full(n_runs, np.nan)

        for run_index in range(n_runs):
            for metric_index, metric in enumerate(self.METRICS):
                medians_per_run[run_index, metric_index] = np.median(
                    samples[run_index][metric.name]
                )

            # Collect the run time (last sample time)
            run_times_per_run[run_index] = samples[run_index]["sample_time"][-1]

        # Compute the median value per metric considering all the runs
        medians = {}
        for metric_index, metric in enumerate(self.METRICS):
            medians[f"median_{metric.name}"] = float(
                np.median(medians_per_run[:, metric_index])
            )
        medians["median_run_time"] = float(np.median(run_times_per_run))

        # Get the index of the run that had the closest run time to the median
        index = np.argmin(np.absolute(run_times_per_run - medians["median_run_time"]))

        return medians, samples[index]

    def __create_plots(
        self, timeline: dict[str, list[float]]
    ) -> matplotlib.figure.Figure:
        """Creates the matplotlib figure with the metrics timeline"""

        n_metrics = len(self.METRICS)
        n_columns = 2
        n_rows = int(np.ceil(n_metrics / n_columns))

        fig, axs = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 3 * n_rows))

        # Plot the results
        flat_axs = axs.flat
        for metric_index, metric in enumerate(self.METRICS):

            # Line
            flat_axs[metric_index].plot(
                timeline["sample_time"], timeline[metric.name], color="blue", zorder=0
            )

            # Sample points
            flat_axs[metric_index].scatter(
                timeline["sample_time"],
                timeline[metric.name],
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
        dict[str, list[float]] (appended to `return_values`)
            Example:
                {
                    GRAPHICS_CLOCK: [1500, 1550, 1600, 1575, 1620],
                    TEMPERATURE: [65, 67, 68, 66, 70],
                    ...
                    sample_time: [ 0, 0.1, 0.2, 0.3, 0.4] # Seconds since the start of the sampling
                }
        """

        period = 1 / self.__NVML_sampling_frequency

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
