"""Tests to verify that MAX graphs properly release the GIL during execution.

This test ensures that multiple MAX graphs can run in parallel Python threads,
which is critical for performance in multi-threaded applications.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from max.graph import Graph, DeviceRef, TensorType
from max.dtype import DType

from alpha_max_zero import kernels
from alpha_max_zero.kernels import sleep


@pytest.fixture
def sleep_graph(cpu_inference_session):
    """Create a graph that sleeps for a specified duration."""
    with Graph(
        "sleep_test",
        input_types=[
            TensorType(dtype=DType.float32, shape=(), device=DeviceRef.CPU()),
        ],
        custom_extensions=[kernels.mojo_kernels],
    ) as graph:
        duration = graph.inputs[0]
        result = sleep(duration)
        graph.output(result)

    return cpu_inference_session.load(graph)


def test_gil_release_parallel_execution(sleep_graph):
    """Test that MAX graphs release the GIL and can run in parallel threads.

    This test runs 4 graphs in parallel, each sleeping for 0.5 seconds.
    If they run in parallel (GIL is released), total time should be ~0.5s.
    If they run sequentially (GIL not released), total time would be ~2.0s.
    """
    num_threads = 4
    sleep_duration = 0.5  # seconds per graph

    def run_graph():
        """Execute the sleep graph and return execution time."""
        start = time.time()
        result = sleep_graph.execute(sleep_duration)[0]
        end = time.time()
        return end - start, result

    # Run graphs in parallel threads
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_graph) for _ in range(num_threads)]

        execution_times = []
        results = []

        for future in as_completed(futures):
            exec_time, result = future.result()
            execution_times.append(exec_time)
            results.append(result)

    total_time = time.time() - start_time

    # Verify all graphs executed successfully
    assert len(results) == num_threads, (
        f"Expected {num_threads} results, got {len(results)}"
    )

    for i, result in enumerate(results):
        result_value = result.to_numpy()
        assert result_value.shape == (1,), (
            f"Graph {i} returned shape {result_value.shape}, expected (1,)"
        )
        assert result_value[0] == 1.0, (
            f"Graph {i} returned {result_value[0]}, expected 1.0"
        )

    # Verify parallel execution
    # If GIL is released: total_time ≈ 0.5s (plus overhead)
    # If GIL not released: total_time ≈ 2.0s
    sequential_time = sleep_duration * num_threads

    # Allow some overhead, but total time should be much less than sequential
    assert total_time < sequential_time * 0.6, (
        f"Graphs appear to run sequentially. "
        f"Total time: {total_time:.2f}s, "
        f"Expected < {sequential_time * 0.6:.2f}s for parallel execution"
    )

    # Also verify that total time is at least the sleep duration
    # (can't be faster than a single sleep)
    assert total_time >= sleep_duration * 0.9, (
        f"Total time {total_time:.2f}s is suspiciously fast, "
        f"expected at least {sleep_duration * 0.9:.2f}s"
    )

    print("\nGIL release test successful!")
    print(f"Ran {num_threads} graphs with {sleep_duration}s sleep each")
    print(
        f"Total time: {total_time:.2f}s (sequential would be ~{sequential_time:.1f}s)"
    )
    print(f"Average speedup: {sequential_time / total_time:.2f}x")
