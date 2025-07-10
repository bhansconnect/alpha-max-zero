import numpy as np
from max.engine import InferenceSession  # pyright: ignore[reportPrivateImportUsage]
from max.graph import Graph

from alpha_max_zero import kernels, game


def test_init():
    with Graph("init", custom_extensions=[kernels.mojo_kernels]) as graph:
        g = game.TicTacToeGame()
        va = g.valid_actions()
        res = g.is_terminal()
        graph.output(va, res)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    results = model.execute()
    valid_actions = results[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    scores = results[1].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    assert all(valid_actions), "All actions should be valid initially"  # pyright: ignore[reportUnknownArgumentType]
    assert not any(scores), "The game should have no results initially"  # pyright: ignore[reportUnknownArgumentType]


def test_winning_game():
    """Test a complete game that results in a win and validate is_terminal."""

    with Graph("winning_game", custom_extensions=[kernels.mojo_kernels]) as graph:
        g = game.TicTacToeGame()
        g.play_action(4)
        g.play_action(0)
        g.play_action(6)
        g.play_action(1)
        g.play_action(2)
        graph.output(g.is_terminal())

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    results = model.execute()[0].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]

    np.testing.assert_array_equal(results, [True, False, False])  # pyright: ignore[reportUnknownArgumentType]
