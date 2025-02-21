from encoded_instruction import EncodedInstruction


class PTXParser:
    """Handles the parsing of a PTX file and instruction encoding"""

    def parse(self, ptx_file: str) -> dict[str : list[EncodedInstruction]]:
        """
        Parses the PTX file and encodes the kernels as a sequence

        Parameters
        ----------
        ptx_file: str
            The PTX file to parse

        Returns
        -------
        dict[str : list[EncodedInstruction]]
            Each entry in the dictionary:
                Key: Name of the kernel
                Value: The sequence of encoded instructions corresponding to the kernel
        """

        kernel_sequences = dict()

        with open(ptx_file, "r") as ptx:
            lines = ptx.readlines()

        # Next part is easier to understand if also looking at a PTX example

        # Remove unnecessary lines
        filtered_lines = []
        for line in lines:
            line = line.strip()

            if (
                # Skip empty lines
                line == ""
                # Skip end of function parameters declaration
                or line == ")"
                # Skip start of kernel instructions (already signalled by the function name line)
                or line == "{"
                # Skip comment lines
                or line.startswith("//")
                # Skip declarations like:
                # .reg .pred 	%p<2>;
                # But don't skip kernel declaration lines like:
                # .visible .entry _Z12simpleKernelIfEvPT_S1_S1_S1_S1_(
                or (line.startswith(".") and not line.endswith("("))
                # Skip labels, like:
                # $L__BB0_2:
                or line.startswith("$")
            ):
                continue

            filtered_lines.append(line)

        current_kernel = None  # Name of the current kernel being processed
        for line in filtered_lines:

            if line.endswith("("):
                # Kernel instructions are going to start being processed
                # Remove kernel name from line like:
                # .visible .entry _Z12simpleKernelIfEvPT_S1_S1_S1_S1_(

                current_kernel = line.split()[-1][:-1]
                kernel_sequences[current_kernel] = list()

            elif line == "}":
                # End of kernel instructions

                current_kernel = None

            else:
                assert current_kernel is not None
                kernel_sequences[current_kernel].append(line)

        return kernel_sequences


if __name__ == "__main__":
    ptx_parser = PTXParser()

    ptx_parser.parse("example.ptx")
