"""Python wrapper for PCG random number generator.

This module provides a high-level interface to the PCG (Permuted Congruential Generator)
random number generator implemented in Mojo as a MAX Graph custom op.
"""

from typing import Union

from max.dtype import DType
from max.graph import (
    _OpaqueType,  # pyright: ignore[reportPrivateUsage]
    _OpaqueValue,  # pyright: ignore[reportPrivateUsage]
    DeviceRef,
    ops,
    TensorType,
    TensorValue,
    ShapeLike,
)


class PCGRandom:
    """PCG random number generator for MAX Graph.

    This class wraps the PCG (Permuted Congruential Generator) random number generator
    implemented in Mojo. The generator state is maintained as an opaque value within
    the MAX Graph, allowing for efficient GPU/accelerator execution.

    Example:
        ```python
        from max.graph import Graph
        from alpha_max_zero.random import PCGRandom

        with Graph("random_example") as graph:
            rng = PCGRandom(seed=42)
            random_tensor = rng.generate_float32((1000,))
            graph.output(random_tensor)
        ```
    """

    value: _OpaqueValue
    """The OpaqueValue representing the PCG state in the graph."""

    def __init__(
        self, seed: Union[int, TensorValue] = 0, stream: Union[int, TensorValue] = 1
    ):
        """Initialize a new PCG random number generator.

        Args:
            seed: Initial seed value. Can be int or TensorValue with dtype uint64 and scalar shape.
            stream: Stream number for independent random sequences. Can be int or TensorValue
                   with dtype uint64 and scalar shape. Each stream produces a different,
                   independent sequence of random numbers.
        """
        if isinstance(seed, int):
            seed = ops.constant(seed, DType.uint64, DeviceRef.CPU())

        if seed.dtype != DType.uint64:
            raise ValueError(f"seed must be uint64, got {seed.dtype}")
        if len(seed.shape) != 0:
            raise ValueError(f"seed must be scalar, got shape {seed.shape}")

        if isinstance(stream, int):
            stream = ops.constant(stream, DType.uint64, DeviceRef.CPU())

        if stream.dtype != DType.uint64:
            raise ValueError(f"stream must be uint64, got {stream.dtype}")
        if len(stream.shape) != 0:
            raise ValueError(f"stream must be scalar, got shape {stream.shape}")

        self.value = ops.custom(
            name="alpha_max_zero.random.pcg.init",
            device=DeviceRef.CPU(),
            values=[seed, stream],
            out_types=[_OpaqueType("PCGState")],
        )[0].opaque

    def seed(self, seed: int) -> None:
        """Re-seed the generator with a new seed value.

        This resets the internal state while keeping the same stream number.
        Useful for ensuring reproducible sequences.

        Args:
            seed: New seed value to use.
        """
        ops.inplace_custom(
            name="alpha_max_zero.random.pcg.seed",
            device=DeviceRef.CPU(),
            values=[
                self.value,
                ops.constant(seed, DType.uint64, DeviceRef.CPU()),
            ],
        )

    def uniform(
        self, low: float = 0.0, high: float = 1.0, shape: ShapeLike = []
    ) -> TensorValue:
        """Generate random numbers from a uniform distribution.

        Args:
            low: Lower bound (inclusive). Default is 0.0.
            high: Upper bound (exclusive). Default is 1.0.
            shape: Shape of the tensor to generate.

        Returns:
            TensorValue with random values uniformly distributed in [low, high).

        Example:
            ```python
            # Generate values between -1 and 1
            random_centered = rng.uniform(-1.0, 1.0, (10,))

            # Generate values between 0 and 10
            random_scaled = rng.uniform(0.0, 10.0, (10,))
            ```
        """
        base_values = ops.inplace_custom(
            name="alpha_max_zero.random.pcg.generate_float32",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(dtype=DType.float32, shape=shape, device=DeviceRef.CPU())
            ],
        )[0].tensor

        # Scale and shift to [low, high)
        scale = high - low
        return base_values * scale + low
