import numpy as np
from max.graph import Graph
from max.driver import Tensor

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

    with Graph("winning_game", custom_extensions=[kernels.mojo_kernels]) as graph:
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
    results = result.to_numpy()

    np.testing.assert_array_equal(results, [True, False, False])  # pyright: ignore[reportUnknownArgumentType]
