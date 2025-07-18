"""Shared pytest fixtures for all tests."""

import pytest
from max.engine import InferenceSession  # pyright: ignore[reportPrivateImportUsage]
from max.driver import CPU

from alpha_max_zero import kernels


@pytest.fixture(scope="session")
def inference_session():
    """Create a configured InferenceSession for testing potentially with an accelerator."""
    session = InferenceSession(devices=[kernels.inference_device])
    session.set_mojo_assert_level("ALL")
    return session


@pytest.fixture()
def cpu_inference_session():
    """Create a configured InferenceSession for testing cpu things that can run in parallel"""
    session = InferenceSession(devices=[CPU()])
    session.set_mojo_assert_level("ALL")
    return session
