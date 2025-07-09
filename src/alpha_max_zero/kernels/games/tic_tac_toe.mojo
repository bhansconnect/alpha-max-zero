"""Actual implementation of tic tac toe in mojo.  
"""
import compiler
from tensor_internal import OutputTensor

from .traits import GameT


@register_passable("trivial")
struct TicTacToeGame(GameT):
    """Super simple game for testing."""

    alias num_players = 2
    alias num_actions = 9

    var board: UInt32
    """The game board in a compressed form.
    Only the bottom 18 bits are used.
    The first 9 represent if a square contains an O.
    The second 9 reresent if a square contains an X.

    The 19th bit indicates turn. 0 for first player. 1 for second player.
    """

    # This is failing mojo format for some reason...
    fn __init__(out self):
        self.board = 0

    fn valid_actions(self, output: OutputTensor[dtype=DType.bool, rank=1]):
        not_board = ~self.board
        free = (not_board >> 9) & not_board

        output[0] = Scalar[DType.bool](free & 0b1_0000_0000)
        output[1] = Scalar[DType.bool](free & 0b0_1000_0000)
        output[2] = Scalar[DType.bool](free & 0b0_0100_0000)
        output[3] = Scalar[DType.bool](free & 0b0_0010_0000)
        output[4] = Scalar[DType.bool](free & 0b0_0001_0000)
        output[5] = Scalar[DType.bool](free & 0b0_0000_1000)
        output[6] = Scalar[DType.bool](free & 0b0_0000_0100)
        output[7] = Scalar[DType.bool](free & 0b0_0000_0010)
        output[8] = Scalar[DType.bool](free & 0b0_0000_0001)

    fn current_player(self) -> Scalar[DType.uint32]:
        return self.board >> 18
    
    fn _already_played(self, action: Scalar[DType.uint32]) -> Bool:
        full = self.board >> 9 | self.board
        action_shift = 8 - action
        position = 1 << action_shift
        return full & position != 0
    
    fn play_action(mut self, action: Scalar[DType.uint32]):
        debug_assert(action < self.num_actions, "action out of range")
        debug_assert(not self._already_played(action), "invalid action, already played")

        player_shift = 9 if self.current_player() == 1 else 0
        action_shift = 8 - action

        position = 1 << (player_shift + action_shift)

        self.board |= position
        self.board ^= 1 << 18


# TODO: look at making one generic entrypoint that is parameterized on the game.
@compiler.register("alpha_max_zero.games.tic_tac_toe.init")
struct Init:
    @always_inline
    @staticmethod
    fn execute() -> TicTacToeGame:
        return TicTacToeGame()
