"""Sleep custom op implementation for testing GIL release."""

import compiler
import time
from tensor_internal import OutputTensor
from utils.index import IndexList

@compiler.register("alpha_max_zero.sleep")
struct Sleep:
    """Sleep for a specified duration and fill output tensor with 1.0.
    
    This custom op is used to test that MAX graphs properly release
    the GIL during execution, allowing true parallel execution in
    Python threads.
    """
    
    @always_inline
    @staticmethod
    fn execute(output: OutputTensor[dtype=DType.float32, rank=1], duration: Scalar[DType.float32]):
        """Sleep for the specified duration and fill output with 1.0.
        
        Args:
            output: Output tensor to fill with 1.0.
            duration: Sleep duration in seconds.
        """
        # Convert to seconds as Float64 for sleep function
        var duration_f64 = Float64(duration)
        time.sleep(duration_f64)
        
        # Fill the entire output tensor with 1.0
        for i in range(output.shape()[0]):
            output[i] = 1.0
