"""Shared pytest fixtures for all tests."""

import pytest
from max.engine import InferenceSession  # pyright: ignore[reportPrivateImportUsage]
from max.driver import CPU

from alpha_max_zero import kernels


@pytest.fixture(scope="session")
def inference_session():
    """Create a configured InferenceSession for testing potentially with an accelerator."""
    return InferenceSession(devices=[kernels.inference_device])


@pytest.fixture()
def cpu_inference_session():
    """Create a configured InferenceSession for testing cpu things that can run in parallel"""
    return InferenceSession(devices=[CPU()])
