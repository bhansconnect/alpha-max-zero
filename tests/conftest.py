"""Shared pytest fixtures for all tests."""

import pytest
from max.engine import InferenceSession  # pyright: ignore[reportPrivateImportUsage]

from alpha_max_zero import kernels


@pytest.fixture
def inference_session():
    """Create a configured InferenceSession for testing."""
    return InferenceSession(devices=[kernels.inference_device])
