from dataclasses import dataclass

import numpy as np

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


@dataclass
class EncodedInstruction:
    """Represents an encoded instruction"""

    instruction_type: InstructionType
    """ The type of the instruction """

    instruction_name: (
        IntegerArithmeticInstructions
        | ExtendedPrecisionIntegerArithmeticInstructions
        | FloatingPointInstructions
        | HalfPrecisionFloatingPointInstructions
        | MixedPrecisionFloatingPointInstructions
        | ComparisonAndSelectionInstructions
        | HalfPrecisionComparisonInstructions
        | LogicAndShiftInstructions
        | DataMovementAndConversionInstructions
        | TextureInstructions
        | SurfaceInstructions
        | ControlFlowInstructions
        | ParallelSynchronizationAndCommunicationInstructions
        | WarpLevelMatrixMultiplyAccumulateInstructions
        | AsynchronousWarpgroupMatrixMultiplyAccumulateInstructions
        | TensorCore5thGenFamilyInstructions
        | StackManipulationInstructions
        | VideoInstructions
        | MiscellaneousInstructions
    )
    """ The instruction itself (type defined by `instruction_type`) """

    state_space: StateSpace
    """ The state (memory) space of the instruction """

    data_type: DataType
    """ The result data type """

    number_of_operands: int
    """ Number of register operands used by the instruction (input + destination) """

    def to_array(self):
        """Transforms the encoded instruction into an array"""

        instruction_type_list = list(InstructionType)
        instructions_list = list(type(self.instruction_name))
        state_space_list = list(StateSpace)
        data_type_list = list(DataType)

        # Use the index of the enums members
        return np.array(
            [
                instruction_type_list.index(self.instruction_type),
                instructions_list.index(self.instruction_name),
                state_space_list.index(self.state_space),
                data_type_list.index(self.data_type),
                self.number_of_operands,
            ]
        )
