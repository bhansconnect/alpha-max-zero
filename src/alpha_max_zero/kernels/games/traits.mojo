"""
The Core traits that all games must implement.
"""

from tensor_internal import OutputTensor


trait GameT(Defaultable, Copyable, Movable):
    """The core trait representing game a playable game."""
    # Limit of 255 players
    alias num_players: UInt8
    # Limit of number of actions to 65,536...
    # I think this is ok for games I care about.
    # Cuts memory usage in two for part of the MCTS though.
    alias num_actions: UInt16

    fn valid_actions(self, output: OutputTensor[dtype=DType.bool, rank=1]):
        """Fill output tensor with valid actions for the current game state."""
        ...

    fn current_player(self) -> Scalar[DType.uint32]:
        """Get the current player (0-based index)."""
        ...

    fn play_action(mut self, action: Scalar[DType.uint32]):
        """Play an action and update the game state."""
        ...

    fn is_terminal(self, results: OutputTensor[dtype=DType.bool, rank=1]):
        """Check if the game has ended.
        
        Outputs:
            - win_status[0 to num_players-1]: True if player N won
            - win_status[num_players]: True if game was a tie.
            - all False means the game is not over.
        """
        ...

