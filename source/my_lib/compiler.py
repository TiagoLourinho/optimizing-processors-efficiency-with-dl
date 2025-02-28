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

    def __init__(self, benchmarks_folder: str):
        self.__benchmarks_folder = benchmarks_folder

    def compile_and_obtain_ptx(self):
        """Obtains the PTX and executable of all the benchmarks composed of only 1 file"""

        # Define neccesary folders to dump the files
        bin_folder = Path(self.__benchmarks_folder) / ".." / "bin"
        ptx_folder = bin_folder / "ptx"
        executables_folder = bin_folder / "executables"

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
                "nvcc",
                *self.NVCC_FLAGS,
                "-ptx",
                str(cu_file_path),
                "-o",
                str(ptx_file),
            ]
            exe_cmd = [
                "nvcc",
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
            f"\n{successfully_compiled}/{total} CUDA files successfully compiled ({success_rate:.2f}%)."
        )


if __name__ == "__main__":
    compiler = Compiler(benchmarks_folder="benchmarks")
    compiler.compile_and_obtain_ptx()
