"""
The Core traits that all games must implement.
"""

from tensor_internal import OutputTensor


trait GameT(Defaultable, Copyable, Movable):
    """The core trait representing game a playable game."""
    alias num_players: UInt8
    alias num_actions: UInt32

    fn valid_actions(self, output: OutputTensor[dtype=DType.bool, rank=1]):
        """Fill output tensor with valid actions for the current game state."""
        ...

    fn current_player(self) -> Scalar[DType.uint32]:
        """Get the current player (0-based index)."""
        ...

    fn play_action(mut self, action: Scalar[DType.uint32]):
        """Play an action and update the game state."""
        ...

    fn is_terminal(self, output: OutputTensor[dtype=DType.bool, rank=1]) -> Bool:
        """Check if the game has ended.
        
        Returns:
            - Bool: True if game is over, False otherwise
            - output[0 to num_players-1]: True if player N won
            - output[num_players]: True if game was a tie.
        """
        ...

