"""Tests of the runtime that cyxla provides: whether the CPU plugin avoids copies."""

import cyxla


def test_runtime_avoids_copies(runtime: cyxla.RuntimeInfo) -> None:
    """Checks that the CPU plugin aliases inputs and exposes outputs in host memory."""
    assert runtime.platform == "cpu"
    assert runtime.aliased_inputs and runtime.inplace_outputs
    assert cyxla.initialize(platform="cpu") == runtime
