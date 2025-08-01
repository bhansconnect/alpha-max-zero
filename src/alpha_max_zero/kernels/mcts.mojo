from memory import UnsafePointer, memcpy
from tensor_internal import InputTensor

from .games.traits import GameT

struct MCTS[G: GameT]:
    """A struct of arrays implementaiton of MCTS.

    The MCTS is specifically Gumbel MCTS with sequential halving.
    The goal is to be cache friendly and generally efficient.
    The MCTS needs to be paired with an evaluator like a neural network.
    """

    var remaining_sims_after_phase: UInt32
    """The number of simulations remaining after the current phase and before needing to pick a node."""

    var remaining_sims_in_phase: UInt32
    """The number of simulations remaining before needing to halve nodes."""

    var max_actions: UInt16
    """The max number of actions to sample at the root node in sequential halving."""

    var halving_nodes: List[UInt32]
    """The nodes currently up for consideration in the sequential halving algorithm."""

    var gumbel_noise: List[Float32]
    """The gumbel noise at the root node for the halving nodes."""

    var size: Int
    """The number of nodes in the tree."""

    var capacity: Int
    """The max number of nodes that could fit in the allocation."""

    # TODO: could this be removed? I think it simplifies things, but is not required.
    var parent_index: UnsafePointer[UInt32]
    """Index of the parent node.

    Root node has self index.
    """

    var pi_logit: UnsafePointer[Float32]
    """The logit results of the action played to reach this node.

    Of note, pi_logit is initialized by the parent node.
    As such, it is always set for nodes.
    """

    var visit_counts: UnsafePointer[UInt32]
    """Number of times an action was played to reach a node.
    
    If the visit count is zero, none of the below fields have been expanded.
    If it is greater than zero, the below fields are initilized.
    """

    # TODO: can this be removed instead of wasting memory?
    # Probably can be calucated.
    var played_action: UnsafePointer[UInt16]
    """The action played to reach a board."""

    # TODO: maybe store a CompactedState.
    # For now, we are just gonna try storing all games.
    # For most games, the states should be small. And this is likely worth it.
    # Technically might be better to use a transposition table, only store the root, and walk from it.
    # Other option is to manually recalculate movement (which is very slow in many games).
    # Also, a transposition solution would allow for node sharing.
    var game_states: UnsafePointer[G]
    """Board state at a given node."""

    var children_index: UnsafePointer[UInt32]
    """Index of first child node.

    If the children index is zero, all the fields below are not yet initilized.
    This is the indicator for node expansion.
    """

    var children_count: UnsafePointer[UInt16]
    """Total number of children for the node."""
    
    # All of the below are attached to the child node memory slot.
    # So Q(S, A) -> This state, child A, q_values result.

    alias WLDArray = InlineArray[Float32, Int(G.num_players + 1)]
    """Win-Loss-Draw array

    Technically it is only that for 2 player games.
    It is win probability for each player followed by draw probability.
    """
    
    var player_values: UnsafePointer[Self.WLDArray]
    """The value of the action played to reach this node.

    Value is per player with an extra node for draws.
    """

    fn __init__(out self, owned root_state: Optional[G]= None):
        self.max_actions = 2
        self.size = 0
        self.capacity = 16

        self.remaining_sims_after_phase = 0
        self.remaining_sims_in_phase = 0
        self.halving_nodes = []
        self.gumbel_noise = []
        
        self.game_states = UnsafePointer[G].alloc(self.capacity)
        self.parent_index = UnsafePointer[UInt32].alloc(self.capacity)
        self.visit_counts = UnsafePointer[UInt32].alloc(self.capacity)
        self.children_index = UnsafePointer[UInt32].alloc(self.capacity)
        self.children_count = UnsafePointer[UInt16].alloc(self.capacity)
        self.played_action = UnsafePointer[UInt16].alloc(self.capacity)
        self.player_values = UnsafePointer[Self.WLDArray].alloc(self.capacity)
        self.pi_logit = UnsafePointer[Float32].alloc(self.capacity)

        self.reset(root_state)


    fn __del__(owned self):
        for i in range(self.size):
            (self.game_states + i).destroy_pointee()

        self.game_states.free()
        self.parent_index.free()
        self.visit_counts.free()
        self.children_index.free()
        self.children_count.free()
        self.played_action.free()
        self.player_values.free()
        self.pi_logit.free()

    fn reset(mut self, owned root_state: Optional[G]= None):
        """Resets the MCTS state while retaining memory capacity."""
        self.remaining_sims_after_phase = 0
        self.remaining_sims_in_phase = 0
        self.halving_nodes.clear()
        self.gumbel_noise.clear()

        for i in range(self.size):
            (self.game_states + i).destroy_pointee()

        self.size = 1
        if root_state:
            self.game_states[0] = root_state.take()
        else:
            self.game_states[0] = G()

        self.parent_index[0] = 0
        self.visit_counts[0] = 0
        
    fn _grow(mut self, new_size: Int):
        # This is the equation taken from the python list.
        self.capacity = new_size 
                + (new_size >> 3)          # + 12.5% of new_size
                + (3 if new_size < 9 else 6)   # +3 for small lists (<9), else +6

        game_states = UnsafePointer[G].alloc(self.capacity)
        parent_index = UnsafePointer[UInt32].alloc(self.capacity)
        visit_counts = UnsafePointer[UInt32].alloc(self.capacity)
        children_index = UnsafePointer[UInt32].alloc(self.capacity)
        children_count = UnsafePointer[UInt16].alloc(self.capacity)
        played_action = UnsafePointer[UInt16].alloc(self.capacity)
        player_values = UnsafePointer[Self.WLDArray].alloc(self.capacity)
        pi_logit = UnsafePointer[Float32].alloc(self.capacity)

        # I think this is safe for game states... this would be a move.
        # That said, I'm not 100% sure in all cases.
        memcpy(game_states, self.game_states, self.size)
        memcpy(parent_index, self.parent_index, self.size)
        memcpy(visit_counts, self.visit_counts, self.size)
        memcpy(children_index, self.children_index, self.size)
        memcpy(children_count, self.children_count, self.size)
        memcpy(played_action, self.played_action, self.size)
        memcpy(player_values, self.player_values, self.size)
        memcpy(pi_logit, self.pi_logit, self.size)

        self.game_states.free()
        self.parent_index.free()
        self.visit_counts.free()
        self.children_index.free()
        self.children_count.free()
        self.played_action.free()
        self.player_values.free()
        self.pi_logit.free()

        self.game_states = game_states
        self.parent_index = parent_index
        self.visit_counts = visit_counts
        self.children_index = children_index
        self.children_count = children_count
        self.played_action = played_action
        self.player_values = player_values
        self.pi_logit = pi_logit

    fn start_search(mut self, sim_count: UInt32, max_actions: UInt16):
        """This is called ones before each search phase to setup the search config."""
        self.remaining_sims_after_phase = sim_count
        self.max_actions = max_actions
        
    
    fn search(mut self) -> List[UInt32]:
        """Continues a search for which action to play.

        Uses Gumbel MCTS with Sequential halving.
        Will return a list of nodes to run evaluations for.
        Returns an empty list if out of searches.

        Update should be called between calls to search with results from the evaluations.
        """
        if self.remaining_sims_after_phase == 0 and self.remaining_sims_in_phase == 0:
            return []

        # Expand root if needed.
        if self.visit_counts[0] == 0:
            # Note: root explicitly does not count as a simulation.
            return [0]

        # Setup next phase with sequential halving if needed.
        if self.remaining_sims_in_phase == 0:
            pass

        # Run a search on each node in phase limited by remaining sims in phase.
        leaves = List[UInt32](capacity = len(self.halving_nodes))
        to_simulate = self.halving_nodes[:Int(self.remaining_sims_in_phase)]
        self.remaining_sims_in_phase -= len(to_simulate)
        for node in to_simulate:
            # find the leaf and append it to the leaves.
            pass
            
        return leaves

    fn update_node(mut self, node: UInt32, policy: InputTensor[dtype=DType.float32, rank=1], result: Self.WLDArray):
        """Update the results for a specific node.

        This will lead to expanding the node.
        For the root node, this will apply gumbel noise.
        """

        # Get valid actions.

        # Add child nodes for each valid action.
        
        # Update action count and propagate value up tree.
        
        if node == 0:
            # Root node: add gumbel noise
            pass

