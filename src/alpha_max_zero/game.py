# This is kinda a crazy idea, and maybe a bad one, but games are Max Graph OpaqueValue.
# This means they are implemented fully in mojo via Max Graph custom ops.

"""
Python wrapper types for Max Graph OpaqueValue games.
The game implementations are in src/alpha_max_zero/games/...
"""

from max.graph import (
    _OpaqueType,  # pyright: ignore[reportPrivateUsage]
    _OpaqueValue,  # pyright: ignore[reportPrivateUsage]
    DeviceRef,
    ops,
)


class Game:
    """Base class for all games managed in the graph.

    Outside the graph, games would be MojoValues instead.
    """

    _name: str
    """Name used for calling custom ops."""

    _type: _OpaqueType
    """The OpaqueType representing this game for custom ops."""

    value: _OpaqueValue
    """The OpaqueValue representing the current game in graph."""

    def __init__(self, name: str, type: _OpaqueType) -> None:
        self._name = name
        self._type = type
        self.value = ops.custom(
            name=f"alpha_max_zero.games.{self._name}.init",
            device=DeviceRef.CPU(),  # TODO: does id matter? Is it core?
            values=[],
            out_types=[self._type],
        )[0].opaque


class TicTacToeGame(Game):
    """Tic tac toe game graph value."""

    def __init__(self) -> None:
        super().__init__("tic_tac_toe", _OpaqueType("TicTacToeGame"))
