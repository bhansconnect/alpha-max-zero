"""The link between python and mojo via max graph custom ops."""

from __future__ import annotations

from pathlib import Path

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, ops, TensorType, TensorValue, Value

mojo_kernels = Path(__file__).parent / "kernels"

inference_device = CPU() if accelerator_count() == 0 else Accelerator()


def sleep(duration: Value | float) -> TensorValue:
    """Sleep for a specified duration and return a constant value.

    This function is primarily used for testing GIL release behavior
    in MAX graphs. It demonstrates that custom ops can release the GIL
    allowing true parallel execution in Python threads.

    Args:
        duration: Sleep duration in seconds (scalar float or tensor)

    Returns:
        TensorValue containing a constant 1.0
    """
    if isinstance(duration, (int, float)):
        duration = ops.constant(float(duration), DType.float32, DeviceRef.CPU())
    assert isinstance(duration, TensorValue)

    if duration.dtype != DType.float32:
        raise ValueError(f"duration must be float32, got {duration.dtype}")
    if len(duration.shape) != 0:
        raise ValueError(f"duration must be scalar, got shape {duration.shape}")

    # Use custom op similar to the PCG pattern
    return ops.custom(
        name="alpha_max_zero.sleep",
        device=DeviceRef.CPU(),
        values=[duration],
        out_types=[TensorType(dtype=DType.float32, shape=(1,), device=DeviceRef.CPU())],
    )[0].tensor
