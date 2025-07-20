Just trying to get a bit more organized as I plan things out.

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
