import random

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import MojoValue  # pyright: ignore[reportPrivateImportUsage]
from max.graph import Graph, TensorType, DeviceRef

from alpha_max_zero import kernels, game


def test_init(cpu_inference_session):
    with Graph("init", custom_extensions=[kernels.mojo_kernels]) as graph:
        g = game.TicTacToeGame()
        va = g.valid_actions()
        res = g.is_terminal()
        graph.output(va, res)

    model = cpu_inference_session.load(graph)

    results = model.execute()
    assert isinstance(results[0], Tensor)
    assert isinstance(results[1], Tensor)
    valid_actions = results[0].to_numpy()
    scores = results[1].to_numpy()

    assert all(valid_actions), "All actions should be valid initially"  # pyright: ignore[reportUnknownArgumentType]
    assert not any(scores), "The game should have no results initially"  # pyright: ignore[reportUnknownArgumentType]


def test_winning_game(cpu_inference_session):
    """Test a complete game that results in a win and validate is_terminal."""

    with Graph("game_ends", custom_extensions=[kernels.mojo_kernels]) as graph:
        g = game.TicTacToeGame()
        g.play_action(4)
        g.play_action(0)
        g.play_action(6)
        g.play_action(1)
        g.play_action(2)
        graph.output(g.is_terminal())

    model = cpu_inference_session.load(graph)

    result = model.execute()[0]
    assert isinstance(result, Tensor)

    np.testing.assert_array_equal(result.to_numpy(), [True, False, False])  # pyright: ignore[reportUnknownArgumentType]


def test_cats_game(cpu_inference_session):
    """Test a complete game that results in a win and validate is_terminal."""

    with Graph("game_ends", custom_extensions=[kernels.mojo_kernels]) as graph:
        g = game.TicTacToeGame()
        g.play_action(4)
        g.play_action(0)
        g.play_action(1)
        g.play_action(7)
        g.play_action(2)
        g.play_action(6)
        g.play_action(3)
        g.play_action(5)
        g.play_action(8)
        graph.output(g.is_terminal())

    model = cpu_inference_session.load(graph)

    result = model.execute()[0]
    assert isinstance(result, Tensor)

    np.testing.assert_array_equal(result.to_numpy(), [False, False, True])  # pyright: ignore[reportUnknownArgumentType]


def test_game_coordination(cpu_inference_session):
    """Plays a few random games making sure graph cordination works"""

    with Graph("init_game", custom_extensions=[kernels.mojo_kernels]) as init_graph:
        init_graph.output(game.TicTacToeGame().value)

    with Graph(
        "play_move",
        input_types=[
            TensorType(dtype=DType.uint32, shape=(), device=DeviceRef.CPU()),
            game.TicTacToeGame.opaque_type(),
        ],
        custom_extensions=[kernels.mojo_kernels],
    ) as action_graph:
        action, g_raw = action_graph.inputs
        g = game.TicTacToeGame(g_raw)
        g.play_action(action)
        action_graph.output(g.valid_actions(), g.is_terminal())

    print(action_graph)
    init = cpu_inference_session.load(init_graph)
    play = cpu_inference_session.load(action_graph)

    # Play a few random games.
    for _ in range(5):
        g = init.execute()[0]
        assert isinstance(g, MojoValue)

        valid, terminal = play(random.randint(0, 8), g)
        assert isinstance(valid, Tensor)
        assert isinstance(terminal, Tensor)

        # First move never ends the game.
        assert not terminal.to_numpy().any()

        while not terminal.to_numpy().any():
            valid_np = valid.to_numpy()
            action = np.random.choice(len(valid_np), p=valid_np / np.sum(valid_np))
            valid, terminal = play(action, g)
            assert isinstance(valid, Tensor)
            assert isinstance(terminal, Tensor)

        assert np.sum(terminal.to_numpy() == 1)
        if terminal.to_numpy()[2]:
            # game was a draw, board must be full
            assert np.sum(valid.to_numpy()) == 0
