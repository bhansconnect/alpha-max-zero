# AlphaMaxZero

## Overview

AlphaMaxZero plans to be an implementation of AlphaZero built using the Max Graph API.
The Max Graph API does not yet have training, so that will have to be hand rolled.
Luckly, AlphaZero works pretty well for small games with very simple networks.

I have already implemented [versions](https://github.com/bhansconnect/alpha_zero_othello) of [alphazero](https://github.com/bhansconnect/fast-alphazero-general) a [few times](https://github.com/bhansconnect/alphazero-pybind11).
Each of my implementations has been signficantly faster than the last.
The previous version, [alphazero-pybind11](https://github.com/bhansconnect/alphazero-pybind11), was more C++ than python.
Hopefully with this repo, the solution can simply be python and mojo.

## Goals

 - Implement a fast single node AlphaZero in Max.
 - Keep as much of the core work in Max and Mojo as possible.
 - Profile and optimize the implementation to maintain high GPU utilization.
 - Implement super basic training via manual backprop.
 - Enable support for multiple games via a simple Mojo trait.
 - Maybe look at multiple GPU or multinode scenarios.

## Potential Techniques

Because I always love to try new techniques, I plan to at least implement [Gumbel MCTS](https://openreview.net/pdf?id=bERaNdoegnO) instead of PUCT MCTS.
I think that with Gumbel MCTS I should be able to get higher batch sizes even when running an individual game.
That said, I still plan to run many games in parallel for maximum batch size (and thus GPU utilization).
I likely will also pull in a handful of the optimization introduced by [KataGo](https://katagotraining.org/) that were also in alphazero-pybind11.
If things go super well, I may eventually also test out a transformer backbone for the network like used by [Leela Chess Zero](https://lczero.org/blog/2024/02/transformer-progress/).

Those are all aspirational goals, but at a minimum, I hope to anchor to DOD where I can and implement as much as possible in mojo with only a thin python wrapper.
This likely means most code will be orchestating Max Graph Custom Ops and passing around OpaqueTypes.
We shall see what the ergonomics of this turns out to be in practice. It definitely will be hitting some potential sharp edges and unstable APIs.
