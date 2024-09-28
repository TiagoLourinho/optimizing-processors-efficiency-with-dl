import os
import subprocess
import threading
import time

from tqdm import tqdm

from .gpu import GPU, GPUQueries


class BenchmarkMonitor:
    """Monitors and samples GPU metrics while executing a CUDA benchmark"""

    SAMPLING_FREQUENCY = 10
    """ The sampling frequency [Hz] """

    def __init__(self, benchmark: str, gpu: GPU, nvcc_path: str, N_runs: str) -> None:

        self.__gpu = gpu
        """ The GPU running the benchmark """

        self.__benchmark = self.__compile(cuda_file=benchmark, nvcc_path=nvcc_path)
        """ The path of the CUDA binary to monitor """

        self.__N_runs = N_runs
        """ The number of times to run the benchmark (to calculate the average results) """

    def __compile(self, cuda_file: str, nvcc_path: str) -> str:
        """Compiles the CUDA program"""

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
        dict (appended to `return_values`)
            Example:
                {
                    GRAPHICS_CLOCK: [1500, 1550, 1600, 1575, 1620],
                    TEMPERATURE: [65, 67, 68, 66, 70],
                    ...
                    sample_time: [ 0, 0.1, 0.2, 0.3, 0.4] # Seconds since the start of the sampling
                }
        """

        metrics = [
            GPUQueries.GRAPHICS_CLOCK,
            GPUQueries.MEMORY_CLOCK,
            GPUQueries.TEMPERATURE,
            GPUQueries.POWER,
            GPUQueries.GPU_UTILIZATION,
        ]

        period = 1 / self.SAMPLING_FREQUENCY

        while not terminate.is_set():

            # Initialize empty samples lists and wait to start
            samples = {metric.name: [] for metric in metrics}
            samples["sample_time"] = []  # Seconds since the start of the sampling
            sample_time = 0

            flag = sample.wait(timeout=0.5)

            if flag:
                while sample.is_set():

                    # Collect samples
                    for metric in metrics:
                        samples[metric.name].append(self.__gpu.query(query_type=metric))

                    samples["sample_time"].append(sample_time)
                    time.sleep(period)
                    sample_time += period

                # "Return" the collected samples
                return_values.append(samples)
                results_ready.set()

    def run_and_monitor(self) -> tuple[dict, dict]:
        """
        Runs the benchmark and monitors it

        Returns
        -------
        tuple[dict, dict]
            - The dictionary with the summary of the main metrics collected and averaged
                Example:
                    {
                        average_GRAPHICS_CLOCK: 1600,
                        average_TEMPERATURE: 67,
                        ...
                        average_run_time: 23
                    }
            - The timeline of the metrics collected that can be plotted
                Example:
                    (see return of __sample_gpu_thread)
        """

        ########## Events ##########

        START_ROI_EVENT = "EVENT:START_ROI"
        END_ROI_EVENT = "EVENT:END_ROI"

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

                if START_ROI_EVENT in stdout_line and not sample.is_set():
                    sample.set()
                elif END_ROI_EVENT in stdout_line and sample.is_set():

                    sample.clear()

                    # Wait for the other thread to append the samples and then collect them
                    results_ready.wait()
                    results.append(return_values[-1])

        ########## Cleanup ##########
        terminate.set()
        sampler_thread.join()

        # TODO: Analyze the results and return the data
        return {}, {}
