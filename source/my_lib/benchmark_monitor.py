import os
import subprocess

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
            else:
                yield output

        if return_code != 0:
            raise Exception(
                f"There was an error running '{' '.join(args)}':\n\n{process.stderr.read().strip()}"
            )

    def run_and_monitor(self) -> tuple[dict, dict]:
        """
        Runs the benchmark and monitors it

        Returns
        -------
        tuple[dict, dict]
            - The dictionary with the summary of the main metrics collected
            - The timeline of the metrics collected that can be plotted
        """

        for _ in tqdm(range(self.__N_runs)):
            for stdout_line in self.__run_command_generator(args=[self.__benchmark]):
                print(stdout_line)
