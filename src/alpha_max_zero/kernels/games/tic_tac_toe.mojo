"""Actual implementation of tic tac toe in mojo.  
"""
import compiler

from .traits import GameT


@register_passable("trivial")
struct TicTacToeGame(GameT):
    """Super simple game for testing."""

    var board: UInt32
    """The game board in a compressed form.
    Only the bottom 18 bits are used.
    The first 9 represent if a square contains an O.
    The second 9 reresent if a square contains an X.
    """

    # This should be `out self`, but that fails to format.
    fn __init__(out self):
        self.board = 0


# TODO: look at making one generic entrypoint that is parameterized on the game.
@compiler.register("alpha_max_zero.games.tic_tac_toe.init")
struct TicTacToeGameInit:
    @always_inline
    @staticmethod
    fn execute() -> TicTacToeGame:
        return TicTacToeGame()
