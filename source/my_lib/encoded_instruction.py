import json
from dataclasses import asdict, dataclass
from enum import Enum

from .PTX_ISA_enums import (
    AsynchronousWarpgroupMatrixMultiplyAccumulateInstructions,
    ComparisonAndSelectionInstructions,
    ControlFlowInstructions,
    DataMovementAndConversionInstructions,
    DataType,
    DependencyType,
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

    kernel_name: str
    """ The kernel this instruction belongs to """

    raw_instruction: str
    """ The raw instruction line """

    instruction_index: int
    """ The index of this instruction on the kernel definition """

    #################### Information encoded on the vector ####################

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

    written_operands: list[str]
    """ Ouputs operands of the instruction (that are written, usually just 1) """

    read_operands: list[str]
    """ Inputs operands of the instruction (that are read) """

    closest_dependency: int
    """ 
    Identifies the offset to the closest instruction that wrote on a register that this instructin reads 
    (-1 for the previous instruction, -2 for two before, etc, -inf for no dependencies) 
    """

    dependecy_type: DependencyType | None
    """ Identifies the type of dependecy, in case it exists """

    is_conditional: bool
    """ Whether or not this encoded instruction is inside a conditional block """

    branching_label: list[str] | None
    """ The branching label this instruction branches to (or None if a normal instruction) """

    branching_offset: int
    """ This attribute represents the branching offset in instructions (0 if it is a normal instruction that doesn't branch) """

    def __str__(self):
        data_dict = asdict(self)
        data_dict["encoded vector"] = self.to_array()
        return json.dumps(data_dict, indent=4, default=str)

    def to_array(self):
        """Transforms the encoded instruction into an array"""

        return [
            self.__get_enum_index(InstructionType, self.instruction_type),
            self.__get_enum_index(type(self.instruction_name), self.instruction_name),
            self.__get_enum_index(StateSpace, self.state_space),
            self.__get_enum_index(DataType, self.data_type),
            len(self.written_operands) + len(self.read_operands),
            self.closest_dependency,
            self.__get_enum_index(DependencyType, self.dependecy_type),
            1 if self.is_conditional else 0,
            self.branching_offset,
        ]

    def __get_enum_index(self, enum: Enum, enum_member: any) -> int:
        """Returns the index of the member of the enum"""

        try:
            return list(enum).index(enum_member)
        except ValueError:
            return -1
