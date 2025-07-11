"""PCG (Permuted Congruential Generator) random number generator implementation.

Based on the minimal C implementation from https://www.pcg-random.org/
"""
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import OutputTensor, foreach
from utils.index import IndexList


@register_passable("trivial")
struct PCGState:
    """PCG random number generator state.
    
    Uses PCG32 algorithm with 64-bit state and 32-bit output.
    """
    
    var state: UInt64
    """Current state of the generator."""
    
    var inc: UInt64
    """Increment (must be odd). Determines the stream."""
    
    fn __init__(out self, seed: UInt64 = 0, stream: UInt64 = 1):
        """Initialize PCG state with seed and stream.
        """
        # Ensure inc is odd by setting low bit
        self.inc = (stream << 1) | 1
        
        # Initialize state and advance once to mix
        self.state = 0
        _ = self._next_uint32()
        self.state += seed
        _ = self._next_uint32()
    
    fn seed(mut self, new_seed: UInt64):
        """Re-seed the generator with a new seed value.
        
        Keeps the same stream (inc value) but resets state.
        """
        # Reset state and re-initialize with new seed
        self.state = 0
        _ = self._next_uint32()
        self.state += new_seed
        _ = self._next_uint32()

    fn _next_uint32(mut self) -> UInt32:
        """Generate next 32-bit random number using PCG algorithm.
        """
        var oldstate = self.state
        
        # Advance internal state using LCG
        # Multiplier from Knuth's MMIX
        self.state = oldstate * 6364136223846793005 + self.inc
        
        # Calculate output function (XSH RR), uses old state for output
        # XorShift: high bits shifted to low, xored with original
        var xorshifted = UInt32(((oldstate >> 18) ^ oldstate) >> 27)
        
        # Random rotation based on high bits
        var rot = UInt32(oldstate >> 59)
        
        # Rotate right
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1) & 31))
    
    fn next_float32(mut self) -> Float32:
        """Generate a random float32 in the range [0, 1).
        """
        # TODO: double check this against the PCG implementation and makes sure it is robust.
        # Convert uint32 to float32 in [0, 1)
        # Multiply by 2^-32
        return Float32(self._next_uint32()) * Float32(2.3283064365386963e-10)
    
    fn generate_float32[rank: Int](mut self, output: OutputTensor[dtype=DType.float32, rank=rank]) raises:
        """Fill a tensor with random float32 values in [0, 1).
        """
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[DType.float32, width]:
            out = SIMD[DType.float32, width]()
            @parameter
            for i in range(width):
                out[i] = self.next_float32()

            return out

        foreach[func](output)

# Custom op registrations

@compiler.register("alpha_max_zero.random.pcg.init")
struct InitPCG:
    """Initialize a new PCG random number generator."""
    
    @always_inline
    @staticmethod
    fn execute(seed: Scalar[DType.uint64], stream: Scalar[DType.uint64]) -> PCGState:
        return PCGState(UInt64(seed), UInt64(stream))


@compiler.register("alpha_max_zero.random.pcg.seed")
struct SeedPCG:
    """Re-seed an existing PCG generator."""
    
    @always_inline
    @staticmethod
    fn execute(mut rng: PCGState, seed: Scalar[DType.uint64]):
        rng.seed(UInt64(seed))


@compiler.register("alpha_max_zero.random.pcg.generate_float32")
struct GenerateFloat32:
    """Generate random float32 values."""
    
    @always_inline
    @staticmethod
    fn execute(output: OutputTensor[dtype=DType.float32, rank=1], mut rng: PCGState) raises:
        rng.generate_float32(output)
