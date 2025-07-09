from max.engine import InferenceSession, MojoValue  # pyright: ignore[reportPrivateImportUsage]
from max.graph import Graph

from alpha_max_zero import kernels, game


def test_init():
    with Graph("init", custom_extensions=[kernels.mojo_kernels]) as graph:
        g = game.TicTacToeGame()
        graph.output(g.value)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    result = model.execute()[0]
    assert isinstance(result, MojoValue)
