just trying to get a bit more organized as I plan things out.

First:
Implement [gumbel MCTS](https://openreview.net/pdf?id=bERaNdoegnO). This will be used for self paly.
I also want to use it for play after time limitted play using [anytime gumbel mcts](https://arxiv.org/pdf/2411.07171) or something similar.
When implementing this, I want the correct base principles:
 - SOA style node storage in flat arrays.
   On updating a move just shift all the live nodes left and remove old nodes keeping capacity
   Not sure if this will work well, but want to try it.
 - Make sure that the mcts can collect many neural net request at once.
   Since gumbel mcts is deterministic per sequential halving phase, should be possible to accumulate many requests even in a single game.
   This should be a huge boon for self play without any form of virtual loss, should allow for more parallelism.
 - Probably no form of graph handling or node information sharing.... Could be a boon, but also adds complications.
 - Caching of network calls should also be possible to skip evaluation.
 - Not sure how to best deal with store state across the searches and being able to resume.
 - For simplicity, might be best to do collect all, prune cached, run uncached, loop.
   Though it would be more parallel to work with cached things ASAP, coordination may be hard.

Make sure to heavily test this.
Should be possible to make situations and see results picking the right thing.

----

This is maybe unnecessarily complex, but I want to test making a way more powerful live MCTS.
As such, I do want to take into account transpositions.

1. each node will store a zobrist hash of the current board (maybe stored directly in the game state?)
2. mcts will use a lookup for child nodes based on zobrist hash. This is extra indirection, when creating nodes.
   That said, after looking up the node, mcts can still just reference an exact child index.
   (maybe use a free list instead of compacting child nodes to beginning of SOA arrays?)
3. Trajectories will track the zobrist hashes to notice repetitions.
   If an mcts rollout hits a repetition allow for either the rollout to be consider a draw or making the move illegal.
4. Games should use incremental zobrist hashes, but we should enable testing against newly computed ones.
5. We will now store edge counts instead of just node counts.
   If the edge count is less than a nodes count, skip rollout, just increment edge count and use the node q value (it is a more accurate estimate).
   If this seems to have issues, just use the MCGS paper node averaging thing instaed.
6. Allow multiple threads to walk a single mcts tree at once.
   They will apply a virtual count to the node while descending the tree.
   This will affect selection of other threads.
   For the root node, allow parallelism within a single phase.
   For root, try to distribute evenly across child nodes.
7. To avoid exploration ruts, in evaluation games with time limits, start with a phase to explore all children nodes (in priority order).
   Limit this phase to x% of total execution time (10%?).
   This phase pushes for at least minimal exploration of as many nodes as possible before going into the deep search.
   Also, use anytime sequential halving for the rest of the execution.
8. Only for selfplay (and batch eval games?), add a nn eval cache.
9. For things like the 25 move rule in tak, only crudely track them in NN input/transposition key.
   Like maybe first 20 moves, do nothing, not 4 set a bit, final one set an extra bit. Then draw.
   This gives some insight without ruining caching.

Note: since the mcts tracks history, it must alert the game of repetitions.
This avoids every single stored game needing to waste space tracking history.


This should get us:
1. Multithreading within a single game. (especially important for faster evals)
2. Transposition based mcts for work sharing.
3. An nn cache for much faster selfplay.

----

This definitely will be a bit hard to implement in a game generic way.
It would be much easier to limit this to a specific bespoke game.
That would be a reasonable downscoping.
but might just try for generic anyway.

----


Next:
A lot of queue infra....

For now, just gonna use threads and try to keep everything in mojo.
That releases the gil and should allow max cpu usage.

No queues pass any data, just integers

mcts threads generate many network requests.
Those can likely be queue via (game index, node index) pairs. maybe array of node indices.
batch worker loads that queue and has the mcts directly write data to a pinned batch tensor
once a batch is full or enough time passes, batch is async sent to gpu and put on queue to execute (again by index)
network thread just hot loops loading the batch queue and then running the network and putting it on the result queue.

Not sure how to best deal with result processing...
instead of dedicated worker, maybe should be done by mcts workers.
Somehow need to manage lifetime of outputs.

For this setup, want one central queue and one batch worker if possible.
This means much higher chance of always having a full batch ready (can pull requests from anywhere)
We obviously at least double buffer the batch, but one central batch means all game simulation work builds up the same single batch.
Warning: if not self play, must batch to two different networks.


When finally running:
anchor to yaml configs.
make it easy to train for any different game by just pointing to a different config.
make it easy to upgrade to new network size by simply pointing to the upgraded config.
make resuming the default or at least a direct question on startup.
