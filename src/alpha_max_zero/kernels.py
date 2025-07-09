
"""The link between python and mojo via max graph custom ops.  
"""

from pathlib import Path

from max.driver import CPU, Accelerator, accelerator_count
from max.graph import DeviceRef, ops

from .game import TicTacToeGame, TicTacToeGameType

mojo_kernels = Path(__file__).parent / "kernels"

inference_device = CPU() if accelerator_count() == 0 else Accelerator()

def tic_tac_toe_init() -> TicTacToeGame:
    return TicTacToeGame(ops.custom(
        name="alpha_max_zero.games.tic_tac_toe.init",
        device = DeviceRef.CPU(), # TODO: does id matter? Is it core?
        values=[],
        out_types=[TicTacToeGameType()],
    )[0].opaque)
