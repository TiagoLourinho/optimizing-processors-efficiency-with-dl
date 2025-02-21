from enum import Enum, auto


class DataType(Enum):
    """Fundamental data types (https://docs.nvidia.com/cuda/parallel-thread-execution/#fundamental-types-fundamental-type-specifiers)"""

    # Signed integer
    S8 = ".s8"
    S16 = ".s16"
    S32 = ".s32"
    S64 = ".s64"

    # Unsigned integer
    U8 = ".u8"
    U16 = ".u16"
    U32 = ".u32"
    U64 = ".u64"

    # Floating-point
    F16 = ".f16"
    F16X2 = ".f16x2"
    F32 = ".f32"
    F64 = ".f64"

    # Bits (untyped)
    B8 = ".b8"
    B16 = ".b16"
    B32 = ".b32"
    B64 = ".b64"
    B128 = ".b128"

    # Predicate
    PRED = ".pred"


class StateSpace(Enum):
    """State Spaces (https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-state-spaces-tab)"""

    REG = ".reg"  # Registers, fast.
    SREG = ".sreg"  # Special registers. Read-only; pre-defined; platform-specific.
    CONST = ".const"  # Shared, read-only memory.
    GLOBAL = ".global"  # Global memory, shared by all threads.
    LOCAL = ".local"  # Local memory, private to each thread.
    PARAM = ".param"  # Kernel parameters, defined per-grid; or function or local parameters, defined per-thread.
    SHARED = ".shared"  # Addressable memory, defined per CTA, accessible to all threads in the cluster throughout the lifetime of the CTA that defines it.
    TEX = ".tex"  # Global texture memory (deprecated).


class InstructionType(Enum):
    """Possible instructions types (https://docs.nvidia.com/cuda/parallel-thread-execution/#instructions)"""

    # Integer Arithmetic Instructions
    INTEGER_ARITHMETIC = auto()

    # Extended-Precision Integer Arithmetic Instructions
    EXTENDED_PRECISION_INTEGER_ARITHMETIC = auto()

    # Floating-Point Instructions
    FLOATING_POINT = auto()

    # Half Precision Floating-Point Instructions
    HALF_PRECISION_FLOATING_POINT = auto()

    # Mixed Precision Floating-Point Instructions
    MIXED_PRECISION_FLOATING_POINT = auto()

    # Comparison and Selection Instructions
    COMPARISON_AND_SELECTION = auto()

    # Half Precision Comparison Instructions
    HALF_PRECISION_COMPARISON = auto()

    # Logic and Shift Instructions
    LOGIC_AND_SHIFT = auto()

    # Data Movement and Conversion Instructions
    DATA_MOVEMENT_AND_CONVERSION = auto()

    # Texture Instructions
    TEXTURE = auto()

    # Surface Instructions
    SURFACE = auto()

    # Control Flow Instructions
    CONTROL_FLOW = auto()

    # Parallel Synchronization and Communication Instructions
    PARALLEL_SYNCHRONIZATION_AND_COMMUNICATION = auto()

    # Warp Level Matrix Multiply-Accumulate Instructions
    WARP_LEVEL_MATRIX_MULTIPLY_ACCUMULATE = auto()

    # Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions
    ASYNCHRONOUS_WARGROUP_MATRIX_MULTIPLY_ACCUMULATE = auto()

    # TensorCore 5th Generation Family Instructions
    TENSORCORE_5TH_GEN_FAMILY = auto()

    # Stack Manipulation Instructions
    STACK_MANIPULATION = auto()

    # Video Instructions
    VIDEO = auto()

    # Miscellaneous Instructions
    MISCELLANEOUS = auto()


class IntegerArithmeticInstructions(Enum):
    """Possible Integer Arithmetic Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions)"""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    MAD = "mad"
    MUL24 = "mul24"
    MAD24 = "mad24"
    SAD = "sad"
    DIV = "div"
    REM = "rem"
    ABS = "abs"
    NEG = "neg"
    MIN = "min"
    MAX = "max"
    POPC = "popc"
    CLZ = "clz"
    BFIND = "bfind"
    FNS = "fns"
    BREV = "brev"
    BFE = "bfe"
    BFI = "bfi"
    BMSK = "bmsk"
    SZEXT = "szext"
    DP4A = "dp4a"
    DP2A = "dp2a"


class ExtendedPrecisionIntegerArithmeticInstructions(Enum):
    """Possible Extended-Precision Integer Arithmetic Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#extended-precision-integer-arithmetic-instructions)"""

    ADD_CC = "add.cc"
    ADDC = "addc"
    SUB_CC = "sub.cc"
    SUBC = "subc"
    MAD_CC = "mad.cc"
    MADC = "madc"


class FloatingPointInstructions(Enum):
    """Possible Floating-Point Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions)"""

    TESTP = "testp"
    COPYSIGN = "copysign"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    FMA = "fma"
    MAD = "mad"
    DIV = "div"
    ABS = "abs"
    NEG = "neg"
    MIN = "min"
    MAX = "max"
    RCP = "rcp"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    SIN = "sin"
    COS = "cos"
    LG2 = "lg2"
    EX2 = "ex2"
    TANH = "tanh"


class HalfPrecisionFloatingPointInstructions(Enum):
    """Possible Half Precision Floating-Point Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions)"""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    FMA = "fma"
    NEG = "neg"
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    TANH = "tanh"
    EX2 = "ex2"


class MixedPrecisionFloatingPointInstructions(Enum):
    """Possible Mixed Precision Floating-Point Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#mixed-precision-floating-point-instructions)"""

    ADD = "add"
    SUB = "sub"
    FMA = "fma"


class ComparisonAndSelectionInstructions(Enum):
    """Possible Comparison and Selection Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions)"""

    SET = "set"
    SETP = "setp"
    SELP = "selp"
    SLCT = "slct"


class HalfPrecisionComparisonInstructions(Enum):
    """Possible Half Precision Comparison Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-comparison-instructions)"""

    SET = "set"
    SETP = "setp"


class LogicAndShiftInstructions(Enum):
    """Possible Logic and Shift Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions)"""

    AND = "and"
    OR = "or"
    XOR = "xor"
    NOT = "not"
    CNOT = "cnot"
    LOP3 = "lop3"
    SHF = "shf"
    SHL = "shl"
    SHR = "shr"


class DataMovementAndConversionInstructions(Enum):
    """Possible Data Movement and Conversion Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions)"""

    MOV = "mov"
    SHFL_SYNC = "shfl.sync"
    PRMT = "prmt"
    LD = "ld"
    LDU = "ldu"
    ST = "st"
    ST_ASYNC = "st.async"
    ST_BULK = "st.bulk"
    MULTIMEM_LD_REDUCE = "multimem.ld_reduce"
    MULTIMEM_ST = "multimem.st"
    MULTIMEM_RED = "multimem.red"
    PREFETCH = "prefetch"
    PREFETCHU = "prefetchu"
    ISSPACEP = "isspacep"
    CVTA = "cvta"
    CVT = "cvt"
    CVT_PACK = "cvt.pack"
    CP_ASYNC = "cp.async"
    CP_ASYNC_COMMIT_GROUP = "cp.async.commit_group"
    CP_ASYNC_WAIT_GROUP = "cp.async.wait_group"
    CP_ASYNC_WAIT_ALL = "cp.async.wait_all"
    CP_ASYNC_BULK = "cp.async.bulk"
    CP_REDUCE_ASYNC_BULK = "cp.reduce.async.bulk"
    CP_ASYNC_BULK_PREFETCH = "cp.async.bulk.prefetch"
    CP_ASYNC_BULK_TENSOR = "cp.async.bulk.tensor"
    CP_REDUCE_ASYNC_BULK_TENSOR = "cp.reduce.async.bulk.tensor"
    CP_ASYNC_BULK_PREFETCH_TENSOR = "cp.async.bulk.prefetch.tensor"
    CP_ASYNC_BULK_COMMIT_GROUP = "cp.async.bulk.commit_group"
    CP_ASYNC_BULK_WAIT_GROUP = "cp.async.bulk.wait_group"
    TENSORMAP_REPLACE = "tensormap.replace"


class TextureInstructions(Enum):
    """Possible Texture Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#texture-instructions)"""

    # Not considered for now as instructions aren't as explicitly defined as the others
    pass


class SurfaceInstructions(Enum):
    """Possible Surface Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#surface-instructions)"""

    SULD = "suld"
    SUST = "sust"
    SURED = "sured"
    SUQ = "suq"


class ControlFlowInstructions(Enum):
    """Possible Control Flow Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions)"""

    BRACES = "{}"
    AT_SYMBOL = "@"
    BRA = "bra"
    CALL = "call"
    RET = "ret"
    EXIT = "exit"


class ParallelSynchronizationAndCommunicationInstructions(Enum):
    """Possible Parallel Synchronization and Communication Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions)"""

    BAR_CTA = "bar{.cta}"
    BARRIER_CTA = "barrier{.cta}"
    BAR_WARP_SYNC = "bar.warp.sync"
    BARRIER_CLUSTER = "barrier.cluster"
    MEMBAR = "membar"
    ATOM = "atom"
    RED = "red"
    RED_ASYNC = "red.async"
    VOTE = "vote"
    MATCH_SYNC = "match.sync"
    ACTIVEMASK = "activemask"
    REDUX_SYNC = "redux.sync"
    GRIDDEPCONTROL = "griddepcontrol"
    ELECT_SYNC = "elect.sync"
    MBARRIER_INIT = "mbarrier.init"
    MBARRIER_INVAL = "mbarrier.inval"
    MBARRIER_ARRIVE = "mbarrier.arrive"
    MBARRIER_ARRIVE_DROP = "mbarrier.arrive_drop"
    MBARRIER_TEST_WAIT = "mbarrier.test_wait"
    MBARRIER_TRY_WAIT = "mbarrier.try_wait"
    MBARRIER_PENDING_COUNT = "mbarrier.pending_count"
    CP_ASYNC_MBARRIER_ARRIVE = "cp.async.mbarrier.arrive"
    TENSORMAP_CP_FENCEPROXY = "tensormap.cp_fenceproxy"
    CLUSTERLAUNCHCONTROL_TRY_CANCEL = "clusterlaunchcontrol.try_cancel"
    CLUSTERLAUNCHCONTROL_QUERY_CANCEL = "clusterlaunchcontrol.query_cancel"


class WarpLevelMatrixMultiplyAccumulateInstructions(Enum):
    """Possible Warp Level Matrix Multiply-Accumulate Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate-instructions)"""

    # Not considered for now as instructions aren't as explicitly defined as the others
    pass


class AsynchronousWarpgroupMatrixMultiplyAccumulateInstructions(Enum):
    """Possible Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)"""

    # Not considered for now as instructions aren't as explicitly defined as the others
    pass


class TensorCore5thGenFamilyInstructions(Enum):
    """Possible TensorCore 5th Generation Family Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-family-instructions)"""

    # Not considered for now as instructions aren't as explicitly defined as the others
    pass


class StackManipulationInstructions(Enum):
    """Possible Stack Manipulation Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions)"""

    STACKSAVE = "stacksave"
    STACKRESTORE = "stackrestore"
    ALLOCA = "alloca"


class VideoInstructions(Enum):
    """Possible Video Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#video-instructions)"""

    VADD = "vadd"
    VADD2 = "vadd2"
    VADD4 = "vadd4"
    VSUB = "vsub"
    VSUB2 = "vsub2"
    VSUB4 = "vsub4"
    VMAD = "vmad"
    VAVRG2 = "vavrg2"
    VAVRG4 = "vavrg4"
    VABSDIFF = "vabsdiff"
    VABSDIFF2 = "vabsdiff2"
    VABSDIFF4 = "vabsdiff4"
    VMIN = "vmin"
    VMIN2 = "vmin2"
    VMIN4 = "vmin4"
    VMAX = "vmax"
    VMAX2 = "vmax2"
    VMAX4 = "vmax4"
    VSHL = "vshl"
    VSHR = "vshr"
    VSET = "vset"
    VSET2 = "vset2"
    VSET4 = "vset4"


class MiscellaneousInstructions(Enum):
    """Possible Miscellaneous Instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions)"""

    BRKPT = "brkpt"
    NANOSLEEP = "nanosleep"
    PMEVENT = "pmevent"
    TRAP = "trap"
    SETMAXNREG = "setmaxnreg"
