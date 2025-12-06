import subprocess
import threading
import time

import numpy as np
import requests

from .gpu import GPU, GPUQueries


SAMPLE = dict[str, dict[str, float]]
""" A sample is a dictionary with the GPU metrics collected at a given time
    
    Example:
        { 
            nvml_metrics: {
                POWER: 100,
                TEMPERATURE: 65,
                ...
            },
            cupti_metrics: {
                gpc__cycles_elapsed.avg.per_second: 1000,
                sys__cycles_elapsed.avg.per_second: 2000,
                ...
            }
        }
"""

RUN_DATA = dict[str, float | list[SAMPLE]]
""" A run data is a dictionary with the GPU metrics collected during a run

    Example:
        {
            runtime: 5, # seconds
            samples: [
                { 
                    nvml_metrics: {
                        POWER: 100,
                        TEMPERATURE: 65,
                        ...
                    },
                    cupti_metrics: {
                        gpc__cycles_elapsed.avg.per_second: 1000,
                        sys__cycles_elapsed.avg.per_second: 2000,
                        ...
                    }
                },
                ...
            ]
        }
"""

PROFILING_SUMMARY = dict[str, float | SAMPLE]
""" A profiling summary is a dictionary representing the average of the metrics collected for a benchmark during all runs

    Example:
        {
            runtime: 5, # seconds
            nvml_metrics: {
                POWER: 100,
                TEMPERATURE: 65,
                ...
            },
            cupti_metrics: {
                gpc__cycles_elapsed.avg.per_second: 1000,
                sys__cycles_elapsed.avg.per_second: 2000,
                ...
            }     
        }
"""


class BenchmarkMonitor:
    """Monitors and samples GPU metrics while executing a CUDA benchmark"""

    __NVML_METRICS_NAMES = [
        GPUQueries.TEMPERATURE,
        GPUQueries.POWER,
        GPUQueries.GPU_UTILIZATION,
    ]
    """ The NVML metrics to collect from the GPU """

    __CUPTI_METRICS_URL = "http://localhost:45535/metrics"
    """ The URL to get CUPTI metrics """

    __CUPTI_SHUTDOWN_URL = "http://localhost:45535/shutdown"
    """ The URL to shutdown CUPTI """

    __CUPTI_EXECUTABLE_PATH = "source/cupti/pm_sampling"
    """ The path to the CUPTI executable """

    ######################################## Dunder methods ########################################

    def __init__(
        self,
        gpu: GPU,
        sampling_frequency: int,
        n_runs: int,
        timeout_seconds: int = None,
    ) -> None:

        self.__gpu = gpu
        """ The GPU running the benchmark """

        #################### Sampling config ####################

        self.__n_runs = n_runs
        """ The number of times to run the benchmark (to calculate the average results) """

        if sampling_frequency is not None:
            self.__sampling_frequency = min(
                sampling_frequency, 100
            )  # Limit sampling frequency to 100 Hz because of NVML
            """ The sampling frequency [Hz] """

        self.__timeout_seconds = timeout_seconds
        """ The timeout for each benchmark run [seconds] """

    def __enter__(self):
        """Starts CUPTI"""

        self.__cupti_process = subprocess.Popen(
            [f"./{self.__CUPTI_EXECUTABLE_PATH}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the CUPTI server to start
        max_retries = 5
        sleep_time = 0.5
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(self.__CUPTI_METRICS_URL, timeout=1)
                if response.status_code == 200:
                    self.__cupti_metrics_names = list(
                        response.json()["metricValues"].keys()
                    )
                    break
            except requests.RequestException:
                pass
            time.sleep(sleep_time)
            retries += 1
        else:
            raise RuntimeError(
                "Failed to connect to CUPTI server after multiple attempts"
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup routines"""

        print("Shutting down CUPTI...")

        # Shutdown CUPTI
        try:
            requests.get(self.__CUPTI_SHUTDOWN_URL)
        except Exception as e:
            print(str(e))
            raise

        # Wait for the CUPTI process to finish
        try:
            self.__cupti_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("CUPTI process did not exit within timeout. Terminating forcefully.")
            self.__cupti_process.terminate()
            try:
                self.__cupti_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("CUPTI process did not terminate. Killing it.")
                self.__cupti_process.kill()

    ################################### Public methods ####################################

    def run_benchmark(
        self, benchmark_path: str, benchmark_args: list
    ) -> PROFILING_SUMMARY:
        """
        Runs the benchmark and monitors it using nvml and cupti

        Parameters
        ----------
        benchmark_path: str
            The benchmark to run
        benchmarks_args: list
            The arguments that should be supplied to the benchmark

        Returns
        -------
        PROFILING_SUMMARY
        """

        ########## Sampler thread management ##########

        # Samples collected by the thread will be appended here
        sampler_output = []
        # Signals when the samples were appended to the return list
        sampler_results_ready_event = threading.Event()
        # Controls when the thread should be sampling
        sample_event = threading.Event()
        # Controls when the thread should terminate
        terminate_event = threading.Event()

        sampler_thread = threading.Thread(
            target=self.__thread_sample_gpu,
            args=(
                sampler_output,
                sampler_results_ready_event,
                sample_event,
                terminate_event,
            ),
        )
        sampler_thread.start()

        try:
            ########## Run benchmark and collect results ##########
            results = []

            for _ in range(self.__n_runs):

                # Clean events and return list before running
                sample_event.clear()
                sampler_results_ready_event.clear()
                sampler_output.clear()

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
                    timeout=self.__timeout_seconds,
                )
                sample_event.clear()

                # Wait for the other thread to append the samples and then collect them
                sampler_results_ready_event.wait()
                results.append(sampler_output[-1])

            ########## Post processing ##########

            terminate_event.set()
            sampler_thread.join()

            return self.__aggregate_all_runs_samples(all_run_data=results)
        finally:
            ########## Cleanup ##########
            terminate_event.set()
            sampler_thread.join()

    def get_nvml_metrics(self) -> dict[str, float]:
        """Returns the NVML metrics collected from the GPU"""
        return {
            metric.name: self.__gpu.query(query_type=metric)
            for metric in self.__NVML_METRICS_NAMES
        }

    def get_cupti_metrics(self) -> dict[str, float]:
        """Returns the CUPTI metrics collected from the GPU"""
        try:
            response = requests.get(self.__CUPTI_METRICS_URL)
            return response.json()[
                "metricValues"
            ]  # timestamps and sample index are not needed
        except Exception as e:
            print(str(e))
            raise

    ################################### Data post processing ####################################

    def __aggregate_all_runs_samples(
        self, all_run_data: list[RUN_DATA]
    ) -> PROFILING_SUMMARY:
        """
        Given all the samples collected for each run,
        calculates the average value per each metric within each run and
        then across runs, yielding the "results summary"

        Parameters
        ----------
        all_run_data: list[RUN_DATA]
            The list of length 'N_runs' containing the output of __thread_sample_gpu method per each run

        Returns
        -------
        PROFILING_SUMMARY
        """

        # Collect all the samples per runs
        all_samples_per_runs = [
            {
                "runtime": None,
                "nvml_metrics": {
                    metric.name: [] for metric in self.__NVML_METRICS_NAMES
                },
                "cupti_metrics": {metric: [] for metric in self.__cupti_metrics_names},
            }
            for _ in range(self.__n_runs)
        ]
        for run_index, run_data in enumerate(all_run_data):
            all_samples_per_runs[run_index]["runtime"] = run_data["runtime"]
            for sample in run_data["samples"]:
                for metric_name, metric_value in sample["nvml_metrics"].items():
                    all_samples_per_runs[run_index]["nvml_metrics"][metric_name].append(
                        metric_value
                    )
                for metric_name, metric_value in sample["cupti_metrics"].items():
                    all_samples_per_runs[run_index]["cupti_metrics"][
                        metric_name
                    ].append(metric_value)

        # Calculate the average value per each metric within each run
        for run_index, run_samples in enumerate(all_samples_per_runs):
            for metric_name, metric_values in run_samples["nvml_metrics"].items():
                run_samples["nvml_metrics"][metric_name] = float(np.mean(metric_values))
            for metric_name, metric_values in run_samples["cupti_metrics"].items():
                run_samples["cupti_metrics"][metric_name] = float(
                    np.mean(metric_values)
                )

        # Calculate the average value per each metric across runs
        profiling_summary = {
            "runtime": 0,
            "nvml_metrics": {metric.name: 0 for metric in self.__NVML_METRICS_NAMES},
            "cupti_metrics": {metric: 0 for metric in self.__cupti_metrics_names},
        }
        for run_samples in all_samples_per_runs:
            profiling_summary["runtime"] += run_samples["runtime"]
            for metric_name, metric_value in run_samples["nvml_metrics"].items():
                profiling_summary["nvml_metrics"][metric_name] += metric_value
            for metric_name, metric_value in run_samples["cupti_metrics"].items():
                profiling_summary["cupti_metrics"][metric_name] += metric_value

        profiling_summary["runtime"] /= self.__n_runs
        for metric_name, metric_value in profiling_summary["nvml_metrics"].items():
            profiling_summary["nvml_metrics"][metric_name] /= self.__n_runs
        for metric_name, metric_value in profiling_summary["cupti_metrics"].items():
            profiling_summary["cupti_metrics"][metric_name] /= self.__n_runs

        return profiling_summary

    ################################### Threaded methods ####################################

    def __thread_sample_gpu(
        self,
        return_values: list,
        results_ready_event: threading.Event,
        sample_event: threading.Event,
        terminate_event: threading.Event,
    ) -> RUN_DATA:
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
        RUN_DATA (appended to `return_values`)
        """

        period = 1 / self.__sampling_frequency

        while not terminate_event.is_set():

            # Initialize empty samples lists and wait to start
            samples = []

            # Use a timeout to avoid blocking forever in case terminate_event is set
            flag = sample_event.wait(timeout=0.5)

            if flag:
                start = time.perf_counter()
                while sample_event.is_set() and not terminate_event.is_set():

                    sample = {
                        "nvml_metrics": self.get_nvml_metrics(),
                        "cupti_metrics": self.get_cupti_metrics(),
                    }
                    samples.append(sample)

                    time.sleep(period)

                end = time.perf_counter()

                # "Return" the collected samples
                run_data = {"runtime": end - start, "samples": samples}
                return_values.append(run_data)
                results_ready_event.set()

    ################################### Others ####################################

    def __wait_to_cooldown(self):
        """Enters a loop that waits for the gpu to cooldown"""

        max_temp = 65  # ºC
        max_util = 10  # %

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
