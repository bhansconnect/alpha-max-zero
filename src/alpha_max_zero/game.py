# This is kinda a crazy idea, and maybe a bad one, but games are Max Graph OpaqueValue.
# This means they are implemented fully in mojo via Max Graph custom ops.

"""
Python wrapper types for Max Graph OpaqueValue games.
The game implementations are in src/alpha_max_zero/games/...
"""

from abc import ABC, abstractmethod

from max.dtype import DType
from max.graph import (
    _OpaqueType,  # pyright: ignore[reportPrivateUsage]
    _OpaqueValue,  # pyright: ignore[reportPrivateUsage]
    DeviceRef,
    ops,
    TensorType,
    TensorValue,
    Value,
)


class Game(ABC):
    """Base class for all games managed in the graph.

    Outside the graph, games would be MojoValues instead.
    """

    value: _OpaqueValue
    """The OpaqueValue representing the current game in graph."""

    @staticmethod
    @abstractmethod
    def custom_op_name() -> str:
        """Returns the name used by this class for custom ops"""
        ...

    @staticmethod
    @abstractmethod
    def opaque_type() -> _OpaqueType:
        """Returns the OpaqueType for the current game in graph."""
        ...

    @staticmethod
    @abstractmethod
    def num_players() -> int:
        """Returns the number of players."""
        ...

    @staticmethod
    @abstractmethod
    def num_actions() -> int:
        """Returns the number of actions possible in the game."""
        ...

    def __init__(self, opaque_value: Value | None = None) -> None:
        if opaque_value:
            assert isinstance(opaque_value, _OpaqueValue)
            self.value = opaque_value
        else:
            self.value = ops.custom(
                name=f"alpha_max_zero.games.{self.custom_op_name()}.init",
                device=DeviceRef.CPU(),
                values=[],
                out_types=[self.opaque_type()],
            )[0].opaque

    def current_player(self) -> TensorValue:
        """Get the current player."""
        return ops.inplace_custom(
            name=f"alpha_max_zero.games.{self.custom_op_name()}.current_player",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(dtype=DType.uint32, shape=(), device=DeviceRef.CPU())
            ],
        )[0].tensor

    def play_action(self, action: Value | int) -> None:
        """Play an action and update the game state."""
        if isinstance(action, int):
            action = ops.constant(action, DType.uint32, DeviceRef.CPU())
        assert isinstance(action, TensorValue)

        ops.inplace_custom(
            name=f"alpha_max_zero.games.{self.custom_op_name()}.play_action",
            device=DeviceRef.CPU(),
            values=[self.value, action],
        )

    def valid_actions(self) -> TensorValue:
        """Get a boolean tensor indicating which actions are valid."""
        return ops.inplace_custom(
            name=f"alpha_max_zero.games.{self.custom_op_name()}.valid_actions",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(
                    dtype=DType.bool,
                    shape=(self.num_actions(),),
                    device=DeviceRef.CPU(),
                )
            ],
        )[0].tensor

    def is_terminal(self) -> TensorValue:
        """Check if the game has ended.

        Returns:
            - First TensorValue: boolean indicating if game is over
            - Second TensorValue: tensor [player0_won, player1_won, ..., is_tie]
        """
        return ops.inplace_custom(
            name=f"alpha_max_zero.games.{self.custom_op_name()}.is_terminal",
            device=DeviceRef.CPU(),
            values=[self.value],
            out_types=[
                TensorType(
                    dtype=DType.bool,
                    shape=(self.num_players() + 1,),
                    device=DeviceRef.CPU(),
                ),
            ],
        )[0].tensor


class TicTacToeGame(Game):
    """Tic tac toe game graph value."""

    @staticmethod
    def custom_op_name() -> str:
        return "tic_tac_toe"

    @staticmethod
    def opaque_type() -> _OpaqueType:
        return _OpaqueType("TicTacToeGame")

    @staticmethod
    def num_players() -> int:
        return 2

    @staticmethod
    def num_actions() -> int:
        return 9
