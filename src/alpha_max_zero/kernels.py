"""The link between python and mojo via max graph custom ops."""

from pathlib import Path

from max.driver import CPU, Accelerator, accelerator_count

mojo_kernels = Path(__file__).parent / "kernels"

inference_device = CPU() if accelerator_count() == 0 else Accelerator()
