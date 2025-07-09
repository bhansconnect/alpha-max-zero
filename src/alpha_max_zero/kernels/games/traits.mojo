"""
The Core traits that all games must implement.
"""


trait GameT(Defaultable, Copyable, Movable):
    """The core trait representing game a playable game."""
    alias num_players: UInt8
    alias num_actions: UInt32

