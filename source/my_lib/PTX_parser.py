from .encoded_instruction import EncodedInstruction
from .PTX_ISA_enums import (
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
    DependencyType,
)


class PTXParser:
    """Handles the parsing of a PTX file and instruction encoding"""

    def parse(
        self, ptx_file: str, convert_to_vectors: bool = False
    ) -> dict[str : list[EncodedInstruction]] | dict[str : list[list]]:
        """
        Parses the PTX file and encodes the kernels as a sequence

        Parameters
        ----------
        ptx_file: str
            The PTX file to parse
        convert_to_vectors: bool = False
            Whether or not to convert the encoded instructions to vectors when returning

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

        # Initial preprocessing
        filtered_lines = []
        i = 0
        while i < len(lines):  # lines list will be changed during the loop
            line = lines[i].strip()

            # Remove comments
            line = line.split("//")[0].rstrip()

            if (
                # Skip empty lines
                line == ""
                # Skip parameters declaration
                or line == "("
                or line == ")"
                # Skip PTX initial declarations
                or line.startswith(".version")
                or line.startswith(".target")
                or line.startswith(".address_size")
                # Skip declarations like:
                # .reg .pred 	%p<2>;
                or (line.startswith(".") and line.endswith(";"))
                # Skip parameters (but not kernel/func declaration lines)
                or (
                    line.startswith(".")
                    and ".entry" not in line
                    and ".func" not in line
                )
                # Skip labels, like:
                # $L__BB0_2:
                or (line.startswith("$") and line.endswith(":"))
            ):
                i += 1
                continue

            # There are some "problematic" lines like:
            # { .reg .b64 %tmp;
            # or
            # cvta.shared.u64 	%rd233, %tmp; }
            # So in these cases separate the {} from the rest of the instruction
            # to avoid later parsing errors and reinsert them in the lines list
            # so they are preprocessed alone in a following iteration
            if line != "{" and line.startswith("{"):
                lines.insert(i + 1, "{")
                lines.insert(i + 2, line[1:].strip())
            elif line != "}" and line.endswith("}"):
                lines.insert(i + 1, line[:-1].strip())
                lines.insert(i + 2, "}")
            else:
                filtered_lines.append(line)

            i += 1

        # The main ideia for the next parsing part
        # is to keep track whether the current line belongs to a kernel
        # or a function or if its like a declaration that can be discarded

        current_kernel = None  # Signals whether inside a kernel or not
        instruction_index = (
            0  # Counts the index of the current instruction being parsed
        )
        last_written = (
            {}
        )  # Keeps track of the instruction that last wrote to a register to check for dependencies

        curly_count = 0  # Counts how many {} blocks were opened (used for kernel declaration and grouped blocks)
        calling_a_function = False  # Signals when some PTX code is calling another function so the parameters lines can be skipped
        for line in filtered_lines:

            # Parsing outside of kernels
            if current_kernel is None and (".entry" in line or ".func" in line):

                # Kernel declaration, grab the kernel name from like:
                # .visible .entry _Z8cgkernelv()
                current_kernel = line.split()[-1].replace("(", "").replace(")", "")
                kernel_sequences[current_kernel] = list()

            # Parsing inside a kernel
            else:

                skip_line = False
                # If the line is just opening/closing with {} just count and skip
                if line == "{":
                    curly_count += 1
                    skip_line = True
                elif line == "}":
                    curly_count -= 1
                    skip_line = True
                # The start of a function calling is like:
                # call.uni (retval0),
                # vprintf,
                # So process and encode the call instruction but then skip the following lines
                # as the instruction is split across multiple lines
                elif line.endswith(",") and not line.startswith(
                    ControlFlowInstructions.CALL.value
                ):
                    calling_a_function = True
                # The end of a function calling is );
                elif line == ");":
                    calling_a_function = False
                    continue  # Still skip this last line even with `calling_a_function` being False

                # Means that the kernel declaration ended (reset vars for next kernel)
                if curly_count == 0 or line == ";":
                    current_kernel = None
                    instruction_index = 0
                    last_written = {}
                    continue
                # Special skipping flags
                elif skip_line or calling_a_function:
                    continue

                ### Proceed to instruction encoding ###

                encoded_instruction = self.__parse_instruction_line(
                    line,
                    kernel_name=current_kernel,
                    instruction_index=instruction_index,
                    last_written=last_written,
                )

                # Update last written to check for dependencies later
                # Contrary to what is done in __parse_instruction_line, here it is not necessary to pay special attention
                # to memory addresses, as in:
                # st.shared.u64 [%r33+8], %rd9;
                # This is because the register itself is not being written, and keeping track of the memory addresses themselves is out of scope
                for operand in encoded_instruction.written_operands:
                    last_written[operand] = encoded_instruction

                instruction_index += 1  # For the next one

                assert (
                    encoded_instruction.instruction_name is not None
                ), f"Couldn't parse '{line}' of kernel {current_kernel} in file {ptx_file}."

                if convert_to_vectors:
                    encoded_instruction = encoded_instruction.to_array()

                kernel_sequences[current_kernel].append(encoded_instruction)

        return kernel_sequences

    def __parse_instruction_line(
        self,
        line: str,
        kernel_name: str,
        instruction_index: int,
        last_written: dict[str:EncodedInstruction],
        is_conditional: bool = False,
    ) -> EncodedInstruction:
        """
        Parses the PTX line and converts it to a encoded instruction

        Parameters
        ----------
        line: str
            The read PTX line with an instruction
        kernel_name: str
            The kernel this instruction belongs to
        instruction_index: int
            The index of the instruction on the kernel definition
        last_written: dict[str:EncodedInstruction]
            Dictionary that contains for each register, the encoded instruction that last wrote to it
        is_conditional: bool = False
            Signals whether or not this line is inside a conditional block

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
        # Mark that but remove the @ part for the next parsing part
        conditional_operand = None
        if parts[0].startswith("@"):
            conditional_operand = parts[0][1:]  # Extract the register
            parts = parts[1:]
            is_conditional = True

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

        ############### Search for closest dependencies ###############

        # Clean: %rd10, %rd233, 16;
        # Remove labels (not operands): @%p1 bra 	$L__BB0_7;
        operands = [
            operand.replace(",", "").replace(";", "")
            for operand in operands
            if "$" not in operand
        ]

        if len(operands) >= 2:
            # Common instructions only produce 1 output but it is made to support more in case it is needed
            written_operands = [operands[0]]
            read_operands = operands[1:]
        else:
            written_operands = []
            read_operands = []

        # On conditional instructions like:
        # @%p1 bra 	$L__BB0_7;
        if conditional_operand is not None:
            read_operands.append(conditional_operand)

        # Check when the operands where last written:
        # If the previous instruction wrote to one of the registers this instruction reads,
        # then the offset is -1, so look for the max starting at -inf (no dependecy)
        max_offset = float("-inf")
        dependency_type = None
        for operand in read_operands:

            # If its a memory address, extract the register/param, examples:
            # [%r33+56] or [_Z14shared_latencyPPyS_iiS_iiii_param_3]
            # As we want to check the last time that register was written
            if operand.startswith("[") and operand.endswith("]"):
                operand = operand[1 : operand.find("+")]

            instruction_that_wrote: EncodedInstruction = last_written.get(operand)
            if instruction_that_wrote is not None:
                neg_offset = (
                    instruction_that_wrote.instruction_index - instruction_index
                )

                if neg_offset > max_offset:
                    max_offset = neg_offset
                    dependency_type = (
                        DependencyType.MEMORY
                        if instruction_that_wrote.instruction_type
                        == InstructionType.DATA_MOVEMENT_AND_CONVERSION
                        else DependencyType.COMPUTATION
                    )

        encoded_instruction = EncodedInstruction(
            kernel_name=kernel_name,
            raw_instruction=line,
            instruction_index=instruction_index,
            instruction_type=instruction_type,
            instruction_name=instruction_name,
            state_space=instruction_state_space,
            data_type=instruction_data_type,
            written_operands=written_operands,
            read_operands=read_operands,
            closest_dependency=max_offset,
            dependecy_type=dependency_type,
            is_conditional=is_conditional,
        )

        return encoded_instruction


if __name__ == "__main__":
    ptx_parser = PTXParser()

    ptx_parser.parse("example.ptx")
