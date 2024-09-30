import os
import subprocess
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .gpu import GPU, GPUQueries


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

    def __init__(
        self,
        benchmark: str,
        gpu: GPU,
        nvcc_path: str,
        N_runs: str,
        sampling_frequency: int,
    ) -> None:

        self.__gpu = gpu
        """ The GPU running the benchmark """

        self.__benchmark = self.__compile(cuda_file=benchmark, nvcc_path=nvcc_path)
        """ The path of the CUDA binary to monitor """

        self.__N_runs = N_runs
        """ The number of times to run the benchmark (to calculate the median results) """

        self.__sampling_frequency = sampling_frequency
        """ The sampling frequency [Hz] """

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

    def __sample_gpu_thread(
        self,
        return_values: list,
        results_ready: threading.Event,
        sample: threading.Event,
        terminate: threading.Event,
    ) -> None:
        """
        Method that should be called in a auxiliar thread to sample the GPU metrics

        Parameters
        ----------
        return_values: list
            A list where the return values (GPU metrics) of this thread will be appended
        results_ready: threading.Event
            An event to signal the main thread when the samples were appended to `return_values` meaning they are ready to be collected
        sample: threading.Event
            An event controlling whether or not this thread should be sampling the GPU
        terminate: threading.Event
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

        period = 1 / self.__sampling_frequency

        while not terminate.is_set():

            # Initialize empty samples lists and wait to start
            samples = {metric.name: [] for metric in self.METRICS}
            samples["sample_time"] = []  # Seconds since the start of the sampling
            sample_time = 0

            flag = sample.wait(timeout=0.5)

            if flag:
                while sample.is_set():

                    # Collect samples
                    for metric in self.METRICS:
                        samples[metric.name].append(self.__gpu.query(query_type=metric))

                    samples["sample_time"].append(sample_time)
                    time.sleep(period)
                    sample_time += period

                # "Return" the collected samples
                return_values.append(samples)
                results_ready.set()

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

    def run_and_monitor(self) -> tuple[dict, dict, matplotlib.figure.Figure]:
        """
        Runs the benchmark and monitors it

        Returns
        -------
        tuple[dict, dict, matplotlib.figure.Figure]
            - dicts -> Check docstring of __process_samples
            - A figure with the execution plots
        """

        ########## Sampler thread management ##########

        return_values = []  # Samples collected by the thread will be appended here
        results_ready = (
            threading.Event()
        )  # Controls when the samples were appended to the return array

        sample = threading.Event()  # Controls when the thread should be sampling
        terminate = threading.Event()  # Controls when the thread shoud terminate

        sampler_thread = threading.Thread(
            target=self.__sample_gpu_thread,
            args=(return_values, results_ready, sample, terminate),
        )
        sampler_thread.start()

        ########## Run benchmark and collect results ##########
        results = []
        print("Running benchmark and collecting samples...")
        for _ in tqdm(range(self.__N_runs)):

            # Clean events and return list before running
            sample.clear()
            results_ready.clear()
            return_values.clear()

            # Check stdout of the benchmark and sample inside the Region Of Interest
            for stdout_line in self.__run_command_generator(args=[self.__benchmark]):

                if self.START_ROI_EVENT in stdout_line and not sample.is_set():
                    sample.set()
                elif self.END_ROI_EVENT in stdout_line and sample.is_set():

                    sample.clear()

                    # Wait for the other thread to append the samples and then collect them
                    results_ready.wait()
                    results.append(return_values[-1])

        ########## Cleanup ##########
        terminate.set()
        sampler_thread.join()

        summary_results, timeline = self.__process_samples(samples=results)
        figure = self.__create_plots(timeline=timeline)

        return summary_results, timeline, figure
