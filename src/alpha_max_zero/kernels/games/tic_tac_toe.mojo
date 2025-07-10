"""Actual implementation of tic tac toe in mojo.  
"""
import compiler
from tensor_internal import OutputTensor, InputTensor
from utils.index import IndexList

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

    fn is_terminal(self, results: OutputTensor[dtype=DType.bool, rank=1]):
        """Check if the game has ended using pure bitwise operations.
        
        Outputs:
            - win_status[0 to num_players-1]: True if player N won
            - win_status[num_players]: True if game was a tie.
            - all False means the game is not over.
        """
        alias win_patterns = [
            0b111_000_000,  # Top row
            0b000_111_000,  # Middle row
            0b000_000_111,  # Bottom row
            0b100_100_100,  # Left column
            0b010_010_010,  # Middle column
            0b001_001_001,  # Right column
            0b100_010_001,  # Main diagonal
            0b001_010_100,  # Anti-diagonal
        ]
        
        player0_board = self.board & 0x1FF
        player1_board = (self.board >> 9) & 0x1FF
        
        var player0_win_bits: UInt32 = 0
        var player1_win_bits: UInt32 = 0
        
        @parameter
        for i in range(8):
            pattern = win_patterns[i]
            
            # Check pattern match: (board & pattern) == pattern
            p0_match = ~((player0_board & pattern) ^ pattern)
            p0_win_bit = UInt32((p0_match & pattern) == pattern)
            player0_win_bits |= p0_win_bit
            
            p1_match = ~((player1_board & pattern) ^ pattern)
            p1_win_bit = UInt32((p1_match & pattern) == pattern)
            player1_win_bits |= p1_win_bit
        
        player0_wins = player0_win_bits != 0
        player1_wins = player1_win_bits != 0
        
        # Game over if someone won or board is full
        full_board = player0_board | player1_board
        all_filled = full_board == 0x1FF
        
        # Tie if board is full and no one won
        is_tie = all_filled & (~player0_wins) & (~player1_wins)
        
        results[0] = Scalar[DType.bool](player0_wins)
        results[1] = Scalar[DType.bool](player1_wins)
        results[2] = Scalar[DType.bool](is_tie)


# Of note, it is likely that most of these will see limited use in python.
# Instead, they will mostly be used directly by the MCTS avoiding python interop.

# TODO: look at making one generic entrypoint that is parameterized on the game.
@compiler.register("alpha_max_zero.games.tic_tac_toe.init")
struct Init:
    @always_inline
    @staticmethod
    fn execute() -> TicTacToeGame:
        return TicTacToeGame()

@compiler.register("alpha_max_zero.games.tic_tac_toe.current_player")
struct CurrentPlayer:
    @always_inline
    @staticmethod
    fn execute(game: TicTacToeGame) -> Scalar[DType.uint32]:
        return game.current_player()

@compiler.register("alpha_max_zero.games.tic_tac_toe.play_action")
struct PlayAction:
    @always_inline
    @staticmethod
    fn execute(mut game: TicTacToeGame, action: Scalar[DType.uint32]):
        game.play_action(action)

@compiler.register("alpha_max_zero.games.tic_tac_toe.valid_actions")
struct ValidActions:
    @always_inline
    @staticmethod
    fn execute(output: OutputTensor[dtype=DType.bool, rank=1], mut game: TicTacToeGame):
        game.valid_actions(output)

@compiler.register("alpha_max_zero.games.tic_tac_toe.is_terminal")
struct IsTerminal:
    @always_inline
    @staticmethod
    fn execute(results: OutputTensor[dtype=DType.bool, rank=1], mut game: TicTacToeGame):
        game.is_terminal(results)
