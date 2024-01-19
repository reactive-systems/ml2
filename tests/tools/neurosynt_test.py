"""NeuroSynt test"""


from ml2.ltl import DecompLTLSpec
from ml2.tools.ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolNeuralLTLSynSolution
from ml2.tools.neurosynt import NeuroSynt
from ml2.tools.spot import SpotAIGERMC


def test_neurosynt_1():
    model_checker = SpotAIGERMC()

    neurosynt = NeuroSynt(
        mc_port=model_checker.port,
        verifier=model_checker.__class__.__name__,
        nvidia_gpus=False,
        start_containerized_service=False,
        start_service=True,
        setup_parameters={
            "batch_size": 1,
            "alpha": 0.5,
            "num_properties": 12,
            "length_properties": 30,
            "beam_size": 1,
            "check_setup": True,
            "model": "ht-50",
        },
    )

    real_spec = DecompLTLSpec.from_dict(
        {
            "inputs": ["r_0", "r_1"],
            "outputs": ["g_0", "g_1"],
            "assumptions": [],
            "guarantees": [
                "(G ((! (g_0)) | (! (g_1))))",
                "(G ((r_0) -> (F (g_0))))",
                "(G ((r_1) -> (F (g_1))))",
            ],
        }
    )
    sol: ToolNeuralLTLSynSolution = neurosynt.synthesize(
        ToolLTLSynProblem(
            parameters={"timeout": 120, "allow_unsound": True},
            specification=real_spec,
            system_format="aiger",
        )
    )

    assert sol.synthesis_solution.status.token() == "realizable"
    assert sol.synthesis_solution.circuit is not None
    assert sol.synthesis_solution.circuit.to_str().startswith("aag")
    assert sol.model_checking_solution is not None
    assert sol.model_checking_solution.status.token() == "satisfied"
