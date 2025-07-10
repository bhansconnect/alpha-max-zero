# This is kinda a crazy idea, and maybe a bad one, but games are Max Graph OpaqueValue.
# This means they are implemented fully in mojo via Max Graph custom ops.

"""
Python wrapper types for Max Graph OpaqueValue games.
The game implementations are in src/alpha_max_zero/games/...
"""

from max.dtype import DType
from max.graph import (
    _OpaqueType,  # pyright: ignore[reportPrivateUsage]
    _OpaqueValue,  # pyright: ignore[reportPrivateUsage]
    DeviceRef,
    ops,
    TensorType,
    TensorValue,
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

    def current_player(self) -> TensorValue:
        """Get the current player (0 or 1)."""
        return ops.inplace_custom(
            name=f"alpha_max_zero.games.{self._name}.current_player",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(dtype=DType.uint32, shape=(), device=DeviceRef.CPU())
            ],
        )[0].tensor

    def play_action(self, action: TensorValue | int) -> None:
        """Play an action and update the game state."""
        if isinstance(action, int):
            action = ops.constant(action, DType.uint32, DeviceRef.CPU())

        ops.inplace_custom(
            name=f"alpha_max_zero.games.{self._name}.play_action",
            device=DeviceRef.CPU(),
            values=[self.value, action],
        )

    def valid_actions(self) -> TensorValue:
        """Get a boolean tensor indicating which actions are valid."""
        return ops.inplace_custom(
            name=f"alpha_max_zero.games.{self._name}.valid_actions",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(dtype=DType.bool, shape=(9,), device=DeviceRef.CPU())
            ],
        )[0].tensor

    def is_terminal(self) -> TensorValue:
        """Check if the game has ended.

        Returns:
            - TensorValue: tensor [player0_won, player1_won, is_tie]
        """
        return ops.inplace_custom(
            name=f"alpha_max_zero.games.{self._name}.is_terminal",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(dtype=DType.bool, shape=(3,), device=DeviceRef.CPU()),
            ],
        )[0].tensor


class TicTacToeGame(Game):
    """Tic tac toe game graph value."""

    def __init__(self) -> None:
        super().__init__("tic_tac_toe", _OpaqueType("TicTacToeGame"))
