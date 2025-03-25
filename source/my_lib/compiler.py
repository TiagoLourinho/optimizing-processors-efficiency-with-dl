import os
import subprocess
from pathlib import Path

from tqdm import tqdm


class Compiler:
    """Handles the compilation and PTX extraction of all the benchmarks"""

    NVCC_FLAGS = [
        f"-gencode=arch=compute_52,code=compute_52",  # Default architecture used by nvcc but explicit
        "-O0",  # No optimization
        "-Xcompiler",
        "-O0",  # No optimization for host compiler
        "-Xptxas",
        "-O0",  # No optimization for PTX assembler
        "-lcuda",
    ]
    """Flags used during compilation"""

    def __init__(self, nvcc_path: str, benchmarks_folder: str):

        self.__nvcc_path: str = nvcc_path
        """ The NVIDIA compiler path """

        self.__benchmarks_folder = benchmarks_folder
        """ The folder containing the benchmarks repos """

    def compile_and_obtain_ptx(self):
        """Obtains the PTX and executable of all the benchmarks composed of only 1 file"""

        # Define necesary folders to dump the files
        bin_folder = Path(self.__benchmarks_folder) / ".." / "bin"
        ptx_folder = bin_folder / "ptx"
        executables_folder = bin_folder / "executables"

        # Check if the folders exist and the benchmarks are compilled
        ptx_files = list(ptx_folder.glob("*")) if ptx_folder.exists() else []
        exe_files = (
            list(executables_folder.glob("*")) if executables_folder.exists() else []
        )

        if (
            ptx_folder.exists()
            and executables_folder.exists()
            and len(ptx_files) > 0
            and len(exe_files) > 0
            and len(ptx_files) == len(exe_files)
        ):

            print(
                f"Skipping compilation as there are already {len(ptx_files)} benchmarks compilled. If compilation is desired, delete the bin folder to trigger it."
            )
            return

        # Create folders if needed
        ptx_folder.mkdir(parents=True, exist_ok=True)
        executables_folder.mkdir(parents=True, exist_ok=True)

        successfully_compiled = 0
        compilation_errors = 0

        # Gather all .cu files inside the benchmarks folder
        cu_files = [
            Path(root) / file
            for root, _, files in os.walk(self.__benchmarks_folder)
            for file in files
            if file.endswith(".cu")
        ]
        assert len(cu_files) > 0, "No benchmark found."

        # Go over all CUDA files and try to compile them
        for cu_file_path in tqdm(cu_files, desc="Compiling CUDA Files", unit="file"):

            relative_path = cu_file_path.relative_to(self.__benchmarks_folder)
            safe_name = str(relative_path).replace("/", "_").replace(" ", "_")

            ptx_file = ptx_folder / f"{safe_name.replace('.cu', '.ptx')}"
            executable_file = executables_folder / f"{safe_name.replace('.cu', '.out')}"

            # NVCC commands
            ptx_cmd = [
                self.__nvcc_path,
                *self.NVCC_FLAGS,
                "-ptx",
                str(cu_file_path),
                "-o",
                str(ptx_file),
            ]
            exe_cmd = [
                self.__nvcc_path,
                *self.NVCC_FLAGS,
                str(cu_file_path),
                "-o",
                str(executable_file),
            ]

            # Try obtaining the PTX and then the executable
            try:
                subprocess.run(
                    ptx_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                subprocess.run(
                    exe_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                successfully_compiled += 1

            except subprocess.CalledProcessError:
                compilation_errors += 1

                # Remove PTX if executable compilation fails
                if ptx_file.exists():
                    ptx_file.unlink()

        total = successfully_compiled + compilation_errors
        success_rate = (successfully_compiled / total) * 100
        print(
            f"\n{successfully_compiled}/{total} CUDA files successfully compiled ({success_rate:.2f}%).\n"
        )


if __name__ == "__main__":
    compiler = Compiler(benchmarks_folder="benchmarks")
    compiler.compile_and_obtain_ptx()
