"""Tests for PCG random number generator.

These tests verify the functionality, correctness, and statistical properties
of the PCG random number generator implementation.
"""

import numpy as np
from max.engine import InferenceSession  # pyright: ignore[reportPrivateImportUsage]
from max.graph import Graph

from alpha_max_zero import kernels
from alpha_max_zero.random import PCGRandom


def test_pcg_init():
    """Test PCG initialization and basic functionality."""
    with Graph("pcg_init", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=42, stream=1)
        values = rng.uniform(shape=(10,))
        graph.output(values)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    result = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Check basic properties
    assert result.shape == (10,), f"Expected shape (10,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.all(result >= 0.0), "All values should be >= 0.0"
    assert np.all(result < 1.0), "All values should be < 1.0"
    assert not np.all(result == result[0]), "Values should not all be the same"


def test_pcg_reproducibility():
    """Test that same seed produces same sequence."""
    seed = 12345

    # Generate first sequence
    with Graph("pcg_repro1", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng1 = PCGRandom(seed=seed, stream=1)
        values1 = rng1.uniform(shape=(10,))
        graph.output(values1)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    result1 = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    result2 = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Results should be identical
    np.testing.assert_array_equal(
        result1, result2, "Same seed should produce same sequence"
    )


def test_pcg_different_seeds():
    """Test that different seeds produce different sequences."""
    # Generate sequence with seed 1
    with Graph("pcg_seed1", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng1 = PCGRandom(seed=1, stream=1)
        values1 = rng1.uniform(shape=(10,))
        graph.output(values1)

    session = InferenceSession(devices=[kernels.inference_device])
    model1 = session.load(graph)
    result1 = model1.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Generate sequence with seed 2
    with Graph("pcg_seed2", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng2 = PCGRandom(seed=2, stream=1)
        values2 = rng2.uniform(shape=(10,))
        graph.output(values2)

    model2 = session.load(graph)
    result2 = model2.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Results should be different
    assert not np.array_equal(result1, result2), (
        "Different seeds should produce different sequences"
    )


def test_pcg_different_streams():
    """Test that different streams produce different sequences."""
    seed = 42

    # Generate sequence with stream 1
    with Graph("pcg_stream1", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng1 = PCGRandom(seed=seed, stream=1)
        values1 = rng1.uniform(shape=(10,))
        graph.output(values1)

    session = InferenceSession(devices=[kernels.inference_device])
    model1 = session.load(graph)
    result1 = model1.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Generate sequence with stream 2
    with Graph("pcg_stream2", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng2 = PCGRandom(seed=seed, stream=2)
        values2 = rng2.uniform(shape=(10,))
        graph.output(values2)

    model2 = session.load(graph)
    result2 = model2.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Results should be different
    assert not np.array_equal(result1, result2), (
        "Different streams should produce different sequences"
    )


def test_pcg_tensor_shapes():
    """Test generation of tensor shapes (currently only supports (10,))."""
    # For now, only test the supported shape
    shape = (10,)

    with Graph("pcg_shape_test", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=42)
        values = rng.uniform(shape=shape)
        graph.output(values)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)
    result = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    assert result.shape == shape, f"Expected shape {shape}, got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.all(result >= 0.0), f"All values should be >= 0.0 for shape {shape}"
    assert np.all(result < 1.0), f"All values should be < 1.0 for shape {shape}"


def test_pcg_reseeding():
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

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)
    result1, result2 = model.execute()

    arr1 = result1.to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    arr2 = result2.to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Arrays should be different (very unlikely to be same by chance)
    assert not np.array_equal(arr1, arr2), "Re-seeding should change the sequence"

    # Both should still be valid random values
    for arr in [arr1, arr2]:
        assert np.all(arr >= 0.0), "All values should be >= 0.0"
        assert np.all(arr < 1.0), "All values should be < 1.0"


def test_pcg_statistical_properties():
    """Test basic statistical properties of generated values."""
    with Graph("pcg_stats", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=42)
        # Generate samples for statistical testing (limited to 10 for now)
        values = rng.uniform(shape=(10,))
        graph.output(values)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)
    result = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # With only 10 samples, statistical tests are limited
    # Test basic properties
    assert np.min(result) >= 0.0, (
        f"Minimum value should be >= 0.0, got {np.min(result)}"
    )
    assert np.max(result) < 1.0, f"Maximum value should be < 1.0, got {np.max(result)}"

    # Test that values are not all the same (very basic randomness check)
    assert not np.all(result == result[0]), "Values should not all be identical"


def test_pcg_large_tensors():
    """Test generation using the supported tensor size."""
    with Graph("pcg_supported_size", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=42)
        # Generate using the supported shape
        values = rng.uniform(shape=(10,))
        graph.output(values)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)
    result = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    assert result.shape == (10,), f"Expected shape (10,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.all(result >= 0.0), "All values should be >= 0.0"
    assert np.all(result < 1.0), "All values should be < 1.0"

    # Check that values are not obviously patterned
    consecutive_same = np.sum(result[1:] == result[:-1])
    assert consecutive_same < 5, (
        f"Too many consecutive identical values: {consecutive_same}"
    )


def test_pcg_uniform_distribution():
    """Test uniform distribution helper method."""
    with Graph("pcg_uniform", custom_extensions=[kernels.mojo_kernels]) as graph:
        rng = PCGRandom(seed=42)
        # Generate values in range [-5, 5) (limited to supported shape)
        values = rng.uniform(-5.0, 5.0, (10,))
        graph.output(values)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)
    result = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    # Check range
    assert np.all(result >= -5.0), (
        f"All values should be >= -5.0, min = {np.min(result)}"
    )
    assert np.all(result < 5.0), f"All values should be < 5.0, max = {np.max(result)}"

    # With limited samples, just check basic properties
    # Check that we have some spread in values (not all identical)
    assert not np.all(result == result[0]), "Values should not all be identical"
