from max.engine.api import InferenceSession
from max.graph import Graph

from alpha_max_zero import kernels


def test_init():
    with Graph("init", custom_extensions=[kernels.mojo_kernels]) as graph:
        game = kernels.tic_tac_toe_init()
        graph.output(game)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    result = model.execute()[0]
    # Yay it runs...but can't do anything with it yet...
    print(result)
