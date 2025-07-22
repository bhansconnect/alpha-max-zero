from memory import UnsafePointer, memcpy

from .games.traits import GameT

struct MCTS[G: GameT]:
    """A struct of arrays implementaiton of MCTS.

    The MCTS is specifically Gumbel MCTS with sequential halving.
    The goal is to be cache friendly and generally efficient.
    The MCTS needs to be paired with an evaluator like a neural network.
    """

    var size: Int
    """The number of nodes in the tree."""

    var capacity: Int
    """The max number of nodes that could fit in the allocation."""

    # TODO: look into a transposition table for:
    # 1. Fast movement from the root to leaf.
    # 2. Looking up neural network evals.
    # Probably should either be per thread or globally shared for this.
    # I think with CAS operations can be lock free with simple bucket updates.
    # Maybe use two different caches to avoid thrashing???
    # Oh, technically could also use it for node reuse. Any non-root node.
    # That said, things like the repetition draw rule in chess,
    # the 25 move draw rule in tak, turn count as network input can all lead to many less hits.
    # Oh, and there could be a special hash based on only data the neural network sees for max hit rate.
    # Maybe start with transposition as the base instead of trees..... hmmm....
    var root_game : G
    """The current game of the root node."""
    
    # TODO: could this be removed? I think it simplifies things, but is not required.
    var parent_index: UnsafePointer[UInt32]
    """Index of the parent node.

    Root node has self index.
    """

    var visit_counts: UnsafePointer[UInt32]
    """Number of times an action was played to reach a node.
    
    If the visit count is zero, none of the below fields have been expanded.
    If it is greater than zero, the below fields are initilized.
    """

    var children_index: UnsafePointer[UInt32]
    """Index of first child node.

    If the children index is zero, all the fields below are not yet initilized.
    This is the indicator for node expansion.
    """

    var children_count: UnsafePointer[UInt16]
    """Total number of children for the node."""
    
    # All of the below are attached to the child node memory slot.
    # So Q(S, A) -> This state, child A, q_values result.

    # TODO: can this be removed instead of wasting memory?
    # Probably can be calucated.
    var played_action: UnsafePointer[UInt16]
    """The action played to reach a board."""

    alias WLDArray = InlineArray[Float32, Int(G.num_players + 1)]
    """Win-Loss-Draw array

    Technically it is only that for 2 player games.
    It is win probability for each player followed by draw probability.
    """
    
    var player_values: UnsafePointer[Self.WLDArray]
    """The value of the action played to reach this node.

    Value is per player with an extra node for draws.
    """

    var pi_logit: UnsafePointer[Float32]
    """The logit results of the action played to reach this node."""

    fn __init__(out self, owned root_state: Optional[G]= None):
        self.size = 0
        self.capacity = 16

        self.parent_index = UnsafePointer[UInt32].alloc(self.capacity)
        self.visit_counts = UnsafePointer[UInt32].alloc(self.capacity)
        self.children_index = UnsafePointer[UInt32].alloc(self.capacity)
        self.children_count = UnsafePointer[UInt16].alloc(self.capacity)
        self.played_action = UnsafePointer[UInt16].alloc(self.capacity)
        self.player_values = UnsafePointer[Self.WLDArray].alloc(self.capacity)
        self.pi_logit = UnsafePointer[Float32].alloc(self.capacity)

        # Default value to avoid uninitialized value complaints, but still share logic with reset.
        self.root_game = G()

        self.reset(root_state)


    fn __del__(owned self):
        self.parent_index.free()
        self.visit_counts.free()
        self.children_index.free()
        self.children_count.free()
        self.played_action.free()
        self.player_values.free()
        self.pi_logit.free()

    fn reset(mut self, owned root_state: Optional[G]= None):
        """Resets the MCTS state while retaining memory capacity."""
        self.size = 1
        if root_state:
            self.root_game = root_state.take()
        else:
            self.root_game = G()

        self.parent_index[0] = 0
        self.visit_counts[0] = 0
        
    fn _grow(mut self, new_size: Int):
        # This is the equation taken from the python list.
        self.capacity = new_size 
                + (new_size >> 3)          # + 12.5% of new_size
                + (3 if new_size < 9 else 6)   # +3 for small lists (<9), else +6

        parent_index = UnsafePointer[UInt32].alloc(self.capacity)
        visit_counts = UnsafePointer[UInt32].alloc(self.capacity)
        children_index = UnsafePointer[UInt32].alloc(self.capacity)
        children_count = UnsafePointer[UInt16].alloc(self.capacity)
        played_action = UnsafePointer[UInt16].alloc(self.capacity)
        player_values = UnsafePointer[Self.WLDArray].alloc(self.capacity)
        pi_logit = UnsafePointer[Float32].alloc(self.capacity)

        memcpy(parent_index, self.parent_index, self.size)
        memcpy(visit_counts, self.visit_counts, self.size)
        memcpy(children_index, self.children_index, self.size)
        memcpy(children_count, self.children_count, self.size)
        memcpy(played_action, self.played_action, self.size)
        memcpy(player_values, self.player_values, self.size)
        memcpy(pi_logit, self.pi_logit, self.size)

        self.parent_index.free()
        self.visit_counts.free()
        self.children_index.free()
        self.children_count.free()
        self.played_action.free()
        self.player_values.free()
        self.pi_logit.free()

        self.parent_index = parent_index
        self.visit_counts = visit_counts
        self.children_index = children_index
        self.children_count = children_count
        self.played_action = played_action
        self.player_values = player_values
        self.pi_logit = pi_logit
