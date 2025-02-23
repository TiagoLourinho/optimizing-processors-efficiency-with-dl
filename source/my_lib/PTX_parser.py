from encoded_instruction import EncodedInstruction
from PTX_ISA_enums import (
    AsynchronousWarpgroupMatrixMultiplyAccumulateInstructions,
    ComparisonAndSelectionInstructions,
    ControlFlowInstructions,
    DataMovementAndConversionInstructions,
    DataType,
    ExtendedPrecisionIntegerArithmeticInstructions,
    FloatingPointInstructions,
    HalfPrecisionComparisonInstructions,
    HalfPrecisionFloatingPointInstructions,
    InstructionType,
    IntegerArithmeticInstructions,
    LogicAndShiftInstructions,
    MiscellaneousInstructions,
    MixedPrecisionFloatingPointInstructions,
    ParallelSynchronizationAndCommunicationInstructions,
    StackManipulationInstructions,
    StateSpace,
    SurfaceInstructions,
    TensorCore5thGenFamilyInstructions,
    TextureInstructions,
    VideoInstructions,
    WarpLevelMatrixMultiplyAccumulateInstructions,
)


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
                kernel_sequences[current_kernel].append(
                    self.__parse_instruction_line(line)
                )

        return kernel_sequences

    def __parse_instruction_line(self, line: str) -> EncodedInstruction:
        """
        Parses the PTX line and converts it to a encoded instruction

        Parameters
        ----------
        line: str
            The read PTX line with an instruction

        Returns
        -------
        EncodedInstruction
            The encoded instruction
        """

        # Line example:
        # mul.f32 	%f20, %f19, 0f3F317218;

        parts = line.split()

        # Conditional lines like:
        # @%p1 bra 	$L__BB2_2;
        # Are not considered in the encoding so remove the @ part
        if parts[0].startswith("@"):
            parts = parts[1:]

        instruction, operands = parts[0], parts[1:]

        ############### Search for the data type ###############

        all_floating_types = {
            DataType.F16,
            DataType.F16X2,
            DataType.F32,
            DataType.F64,
        }

        instruction_data_type = None
        mixed_counter = 0
        for current_data_type in DataType:
            if current_data_type.value in instruction:
                instruction_data_type = current_data_type

                # Look for mixed precision instructions
                if current_data_type in all_floating_types:
                    mixed_counter += 1

        if instruction_data_type is not None:
            is_floating_point = instruction_data_type in all_floating_types

            is_mixed_precision = is_floating_point and mixed_counter > 1

            is_half_precision = (
                is_floating_point
                and not is_mixed_precision
                and instruction_data_type
                in {
                    DataType.F16,
                    DataType.F16X2,
                }
            )

        ############### Search for the state space ###############

        instruction_state_space = None
        for current_state_space in StateSpace:
            if current_state_space.value in instruction:
                instruction_state_space = current_state_space
                break

        ############### Search for the instruction type and name ###############

        to_search_instructions_enums = {
            InstructionType.INTEGER_ARITHMETIC: IntegerArithmeticInstructions,
            InstructionType.EXTENDED_PRECISION_INTEGER_ARITHMETIC: ExtendedPrecisionIntegerArithmeticInstructions,
            InstructionType.FLOATING_POINT: FloatingPointInstructions,
            InstructionType.HALF_PRECISION_FLOATING_POINT: HalfPrecisionFloatingPointInstructions,
            InstructionType.MIXED_PRECISION_FLOATING_POINT: MixedPrecisionFloatingPointInstructions,
            InstructionType.COMPARISON_AND_SELECTION: ComparisonAndSelectionInstructions,
            InstructionType.HALF_PRECISION_COMPARISON: HalfPrecisionComparisonInstructions,
            InstructionType.LOGIC_AND_SHIFT: LogicAndShiftInstructions,
            InstructionType.DATA_MOVEMENT_AND_CONVERSION: DataMovementAndConversionInstructions,
            InstructionType.TEXTURE: TextureInstructions,
            InstructionType.SURFACE: SurfaceInstructions,
            InstructionType.CONTROL_FLOW: ControlFlowInstructions,
            InstructionType.PARALLEL_SYNCHRONIZATION_AND_COMMUNICATION: ParallelSynchronizationAndCommunicationInstructions,
            InstructionType.WARP_LEVEL_MATRIX_MULTIPLY_ACCUMULATE: WarpLevelMatrixMultiplyAccumulateInstructions,
            InstructionType.ASYNCHRONOUS_WARGROUP_MATRIX_MULTIPLY_ACCUMULATE: AsynchronousWarpgroupMatrixMultiplyAccumulateInstructions,
            InstructionType.TENSORCORE_5TH_GEN_FAMILY: TensorCore5thGenFamilyInstructions,
            InstructionType.STACK_MANIPULATION: StackManipulationInstructions,
            InstructionType.VIDEO: VideoInstructions,
            InstructionType.MISCELLANEOUS: MiscellaneousInstructions,
        }

        # Some of the instructions sets have equal names (like add for integer and floating point)
        # So they need to be removed accordingly depending on the data type
        if instruction_data_type is not None:
            if is_floating_point:
                del to_search_instructions_enums[InstructionType.INTEGER_ARITHMETIC]

                if is_mixed_precision:
                    del to_search_instructions_enums[InstructionType.FLOATING_POINT]
                    del to_search_instructions_enums[
                        InstructionType.HALF_PRECISION_FLOATING_POINT
                    ]
                    del to_search_instructions_enums[
                        InstructionType.HALF_PRECISION_COMPARISON
                    ]
                else:
                    del to_search_instructions_enums[
                        InstructionType.MIXED_PRECISION_FLOATING_POINT
                    ]

                    if is_half_precision:
                        del to_search_instructions_enums[InstructionType.FLOATING_POINT]
                        del to_search_instructions_enums[
                            InstructionType.COMPARISON_AND_SELECTION
                        ]
                    else:
                        del to_search_instructions_enums[
                            InstructionType.HALF_PRECISION_FLOATING_POINT
                        ]
                        del to_search_instructions_enums[
                            InstructionType.HALF_PRECISION_COMPARISON
                        ]
            else:
                del to_search_instructions_enums[InstructionType.FLOATING_POINT]
                del to_search_instructions_enums[
                    InstructionType.MIXED_PRECISION_FLOATING_POINT
                ]
                del to_search_instructions_enums[
                    InstructionType.HALF_PRECISION_FLOATING_POINT
                ]
                del to_search_instructions_enums[
                    InstructionType.HALF_PRECISION_COMPARISON
                ]

        # Search all the enums with the possible instructions to find it
        instruction_type = None
        instruction_name = None
        for (
            current_instruction_type,
            current_instruction_enum,
        ) in to_search_instructions_enums.items():
            for current_instruction_name in current_instruction_enum:

                if instruction.startswith(current_instruction_name.value):

                    if instruction_name is not None:
                        assert (
                            current_instruction_name.value != instruction_name.value
                        ), f"{current_instruction_name.value} appears more than once in the available instructions."

                    # Fix corner case of instructions that are contained in another instruction
                    # like "set" and "setp"
                    # by picking the longest one
                    if instruction_name is None or len(
                        current_instruction_name.value
                    ) > len(instruction_name.value):
                        instruction_type = current_instruction_type
                        instruction_name = current_instruction_name

        ############### Search for the number of operands ###############

        instruction_number_of_operands = len(operands)

        encoded_instruction = EncodedInstruction(
            instruction_type=instruction_type,
            instruction_name=instruction_name,
            state_space=instruction_state_space,
            data_type=instruction_data_type,
            number_of_operands=instruction_number_of_operands,
        )

        return encoded_instruction


if __name__ == "__main__":
    ptx_parser = PTXParser()

    ptx_parser.parse("example.ptx")
