# This is kinda a crazy idea, and maybe a bad one, but games are Max Graph OpaqueValue.
# This means they are implemented fully in mojo via Max Graph custom ops.

"""
Python wrapper types for Max Graph OpaqueValue games.
The game implementations are in src/alpha_max_zero/games/...
"""

from max.graph import (
    _OpaqueType,  # pyright: ignore[reportPrivateUsage]
    _OpaqueValue,  # pyright: ignore[reportPrivateUsage]
)


class TicTacToeGameType(_OpaqueType):
    """Tic tac toe game graph type."""

    def __init__(self) -> None:
        super().__init__("TicTacToeGame")


class TicTacToeGame(_OpaqueValue):
    """Tic tac toe game graph value."""
