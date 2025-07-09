
from max.engine import InferenceSession, MojoValue
from max.graph import Graph

from alpha_max_zero import kernels
from alpha_max_zero.game import TicTacToeGame

def test_init():
    with Graph("init", custom_extensions=[kernels.mojo_kernels]) as graph:
        game = kernels.tic_tac_toe_init()
        graph.output(game)

    session = InferenceSession(devices=[kernels.inference_device])
    model = session.load(graph)

    result = model.execute()[0]
    assert(isinstance(result, MojoValue))
