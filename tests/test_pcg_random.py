"""Tests for PCG random number generator.

These tests verify the functionality, correctness, and statistical properties
of the PCG random number generator implementation.
"""

import numpy as np
import pytest
from max.graph import Graph, TensorType, DeviceRef
from max.driver import Tensor
from max.dtype import DType

from alpha_max_zero import kernels
from alpha_max_zero.random import PCGRandom


# This uses the global inference_session to enable it to be a modlue scope fixture.
# This leads to more caching and saves execution time.
@pytest.fixture(scope="module")
def random_graph(inference_session):
    seed_type = TensorType(dtype=DType.uint64, shape=(), device=DeviceRef.CPU())
    stream_type = TensorType(dtype=DType.uint64, shape=(), device=DeviceRef.CPU())

    with Graph(
        "pcg_random",
        input_types=(seed_type, stream_type),
        custom_extensions=[kernels.mojo_kernels],
    ) as graph:
        seed_input, stream_input = graph.inputs

        rng = PCGRandom(seed=seed_input.tensor, stream=stream_input.tensor)
        values = rng.uniform(shape=(10,))
        graph.output(values)

    return inference_session.load(graph)


def test_pcg_functionality(random_graph):
    """Test PCG initialization, basic functionality, and reproducibility."""
    seed_tensor = Tensor.scalar(42, DType.uint64)
    stream_tensor = Tensor.scalar(1, DType.uint64)

    # Test basic functionality
    result = random_graph.execute(seed_tensor, stream_tensor)[0]
    assert isinstance(result, Tensor)
    result = result.to_numpy()

    # Check basic properties
    assert result.shape == (10,), f"Expected shape (10,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.all(result >= 0.0), "All values should be >= 0.0"
    assert np.all(result < 1.0), "All values should be < 1.0"
    assert not np.all(result == result[0]), "Values should not all be the same"

    # Test reproducibility - same seed should produce same sequence
    result1 = random_graph.execute(seed_tensor, stream_tensor)[0]
    assert isinstance(result1, Tensor)
    result1 = result1.to_numpy()

    result2 = random_graph.execute(seed_tensor, stream_tensor)[0]
    assert isinstance(result2, Tensor)
    result2 = result2.to_numpy()

    np.testing.assert_array_equal(
        result1, result2, "Same seed should produce same sequence"
    )


def test_pcg_different_seeds(random_graph):
    """Test that different seeds produce different sequences."""
    seed1_tensor = Tensor.scalar(1, DType.uint64)
    seed2_tensor = Tensor.scalar(2, DType.uint64)
    stream_tensor = Tensor.scalar(1, DType.uint64)

    result1 = random_graph.execute(seed1_tensor, stream_tensor)[0]
    assert isinstance(result1, Tensor)
    result1 = result1.to_numpy()

    result2 = random_graph.execute(seed2_tensor, stream_tensor)[0]
    assert isinstance(result2, Tensor)
    result2 = result2.to_numpy()

    assert not np.array_equal(result1, result2), (
        "Different seeds should produce different sequences"
    )


def test_pcg_different_streams(random_graph):
    """Test that different streams produce different sequences."""
    seed = 42
    seed_tensor = Tensor.scalar(seed, DType.uint64)
    stream1_tensor = Tensor.scalar(1, DType.uint64)
    stream2_tensor = Tensor.scalar(2, DType.uint64)

    result1 = random_graph.execute(seed_tensor, stream1_tensor)[0]
    assert isinstance(result1, Tensor)
    result1 = result1.to_numpy()

    result2 = random_graph.execute(seed_tensor, stream2_tensor)[0]
    assert isinstance(result2, Tensor)
    result2 = result2.to_numpy()

    assert not np.array_equal(result1, result2), (
        "Different streams should produce different sequences"
    )


def test_pcg_reseeding(cpu_inference_session):
    """Test re-seeding functionality."""
    initial_seed = 100
    new_seed = 200

    with Graph("pcg_reseed", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=initial_seed, stream=1)

        # Generate some values
        values1 = rng.uniform(shape=(10,))

        # Re-seed
        rng.seed(new_seed)

        # Generate more values
        values2 = rng.uniform(shape=(10,))

        graph.output(values1, values2)

    model = cpu_inference_session.load(graph)
    result1, result2 = model.execute()

    assert isinstance(result1, Tensor)
    assert isinstance(result2, Tensor)
    arr1 = result1.to_numpy()
    arr2 = result2.to_numpy()

    assert not np.array_equal(arr1, arr2), "Re-seeding should change the sequence"


def test_pcg_large_tensors_and_statistics(cpu_inference_session):
    """Test generation of large tensors (1000 elements) and statistical properties."""
    # Create a large tensor graph
    seed_type = TensorType(dtype=DType.uint64, shape=(), device=DeviceRef.CPU())
    stream_type = TensorType(dtype=DType.uint64, shape=(), device=DeviceRef.CPU())

    with Graph(
        "pcg_large_random",
        input_types=(seed_type, stream_type),
        custom_extensions=[kernels.mojo_kernels],
    ) as graph:
        seed_input, stream_input = graph.inputs
        rng = PCGRandom(seed=seed_input.tensor, stream=stream_input.tensor)
        values = rng.uniform(shape=(1000,))
        graph.output(values)

    model = cpu_inference_session.load(graph)

    seed_tensor = Tensor.scalar(42, DType.uint64)
    stream_tensor = Tensor.scalar(1, DType.uint64)
    result = model.execute(seed_tensor, stream_tensor)[0]
    assert isinstance(result, Tensor)
    result = result.to_numpy()

    # Test shape and basic properties
    assert result.shape == (1000,), f"Expected shape (1000,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.all(result >= 0.0), "All values should be >= 0.0"
    assert np.all(result < 1.0), "All values should be < 1.0"

    # Check that values are not obviously patterned
    consecutive_same = np.sum(result[1:] == result[:-1])
    assert consecutive_same < 50, (  # Allow more with larger sample size
        f"Too many consecutive identical values: {consecutive_same}"
    )

    # Test statistical properties with larger sample
    mean = np.mean(result)
    assert 0.4 < mean < 0.6, f"Mean should be near 0.5, got {mean}"

    # Test that we have good spread across the range
    assert np.min(result) < 0.1, "Should have some values near 0"
    assert np.max(result) > 0.9, "Should have some values near 1"

    # Test that values are not all the same (randomness check)
    assert not np.all(result == result[0]), "Values should not all be identical"


def test_pcg_uniform_distribution(cpu_inference_session):
    """Test uniform distribution helper method."""
    with Graph("pcg_uniform", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=42)
        # Generate values in range [-5, 5) (limited to supported shape)
        values = rng.uniform(-5.0, 5.0, (10,))
        graph.output(values)

    model = cpu_inference_session.load(graph)
    result = model.execute()[0]
    assert isinstance(result, Tensor)
    result = result.to_numpy()

    # Check range
    assert np.all(result >= -5.0), (
        f"All values should be >= -5.0, min = {np.min(result)}"
    )
    assert np.all(result < 5.0), f"All values should be < 5.0, max = {np.max(result)}"

    # With limited samples, just check basic properties
    # Check that we have some spread in values (not all identical)
    assert not np.all(result == result[0]), "Values should not all be identical"
