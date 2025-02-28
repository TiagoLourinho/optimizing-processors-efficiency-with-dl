from dataclasses import dataclass
from enum import Enum

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

    state_space: StateSpace | None
    """ The state (memory) space of the instruction """

    data_type: DataType | None
    """ The result data type """

    number_of_operands: int
    """ Number of register operands used by the instruction (input + destination) """

    is_conditional: bool
    """ Whether or not this encoded instruction is inside a conditional block """

    def to_array(self):
        """Transforms the encoded instruction into an array"""

        return [
            self.__get_enum_index(InstructionType, self.instruction_type),
            self.__get_enum_index(type(self.instruction_name), self.instruction_name),
            self.__get_enum_index(StateSpace, self.state_space),
            self.__get_enum_index(DataType, self.data_type),
            self.number_of_operands,
            1 if self.is_conditional else 0,
        ]

    def __get_enum_index(self, enum: Enum, enum_member: any) -> int:
        """Returns the index of the member of the enum"""

        try:
            return list(enum).index(enum_member)
        except ValueError:
            return -1


if __name__ == "__main__":
    inst = EncodedInstruction(
        instruction_type=InstructionType.FLOATING_POINT,
        instruction_name=FloatingPointInstructions.COS,
        state_space=None,
        data_type=DataType.F32,
        number_of_operands=2,
    )

    print(inst.to_array())
