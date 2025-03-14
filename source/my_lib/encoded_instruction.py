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

# Dictionary mapping InstructionType to its respective Enum class
INSTRUCTION_ENUM_MAP = {
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

# Assign a unique global index to each instruction across all enums
global_index_counter = 0
GLOBAL_INSTRUCTION_INDEX = {}

for enum in INSTRUCTION_ENUM_MAP.values():
    for member in list(enum):
        GLOBAL_INSTRUCTION_INDEX[member] = global_index_counter
        global_index_counter += 1


@dataclass
class EncodedInstruction:
    """Represents an encoded instruction"""

    kernel_name: str
    """ The kernel this instruction belongs to """

    raw_instruction: str
    """ The raw instruction line """

    instruction_index: int
    """ The index of this instruction on the kernel definition """

    branching_label: str | None
    """ The branching label this instruction branches to (or None if a normal instruction) """

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

    state_space: StateSpace
    """ The state (memory) space of the instruction """

    data_type: DataType
    """ The result data type """

    written_operands: list[str]
    """ Outputs operands of the instruction (that are written, usually just 1) """

    read_operands: list[str]
    """ Inputs operands of the instruction (that are read) """

    closest_dependency: int
    """ 
    Identifies the offset to the closest instruction that wrote on a register that this instruction reads 
    (-1 for the previous instruction, -2 for two before, etc, -inf for no dependencies) 
    """

    dependecy_type: DependencyType
    """ Identifies the type of dependency, in case it exists """

    is_conditional: bool
    """ Whether or not this encoded instruction is inside a conditional block """

    branching_offset: int
    """ This attribute represents the branching offset in instructions (0 if it is a normal instruction that doesn't branch) """

    def __str__(self):
        data_dict = asdict(self)
        data_dict["encoded vector"] = self.to_dict()
        return json.dumps(data_dict, indent=4, default=str)

    def to_dict(self):
        """Transforms the encoded instruction into a dictionary separating the categorical and numerical features"""

        return {
            "categorical": [
                self.__get_enum_index(InstructionType, self.instruction_type),
                GLOBAL_INSTRUCTION_INDEX[
                    self.instruction_name
                ],  # Use the global index as it can be from multiple enums
                self.__get_enum_index(StateSpace, self.state_space),
                self.__get_enum_index(DataType, self.data_type),
                self.__get_enum_index(DependencyType, self.dependecy_type),
            ],
            "numerical": [
                len(self.written_operands),
                len(self.read_operands),
                self.closest_dependency,
                int(self.is_conditional),
                self.branching_offset,
            ],
        }

    def __get_enum_index(self, enum: Enum, enum_member: any) -> int:
        """Returns the index of the member of the enum"""

        return list(enum).index(enum_member)

    @classmethod
    def get_enconding_info(cls):
        """Returns the categorical sizes of the enums and the number of numerical features"""

        # Should match the order and features defined in to_dict
        return {
            "categorical_sizes": [
                len(InstructionType),
                len(GLOBAL_INSTRUCTION_INDEX),
                len(StateSpace),
                len(DataType),
                len(DependencyType),
            ],
            "n_numerical_features": 5,
        }
