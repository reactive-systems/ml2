"""NeuroSynt"""

import json
import logging
from datetime import timedelta
from typing import Any, Dict, Generator, Tuple

from grpc._channel import _InactiveRpcError

from ...globals import CONTAINER_REGISTRY
from ...grpc.neurosynt import neurosynt_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_syn.ltl_syn_status import LTLSynStatus
from ...verifier import Verifier
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_syn_problem import (
    ToolLTLSynProblem,
    ToolLTLSynSolution,
    ToolNeuralLTLSynSolution,
)
from .neurosynt_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEUROSYNT_IMAGE_NAME_CPU = CONTAINER_REGISTRY + "/neurosynt-grpc-server:cpu"
NEUROSYNT_IMAGE_NAME_GPU = CONTAINER_REGISTRY + "/neurosynt-grpc-server:gpu"


class NeuroSynt(GRPCService):
    def __init__(
        self,
        image: str = None,
        service=serve,
        cpu_count: int = 2,
        mem_limit: str = "10g",
        mc_port: int = 50052,
        verifier: str = "NuxmvMC",
        nvidia_gpus: bool = False,
        setup_parameters: Dict[str, Any] = {},
        **kwargs,
    ):
        if image is None:
            if nvidia_gpus:
                image = NEUROSYNT_IMAGE_NAME_GPU
            else:
                image = NEUROSYNT_IMAGE_NAME_CPU
        super().__init__(
            image=image,
            service=service,
            cpu_count=cpu_count,
            mem_limit=mem_limit,
            tool="NeuroSynt",
            nvidia_gpus=nvidia_gpus,
            network_mode="host",
            stub=neurosynt_pb2_grpc.NeuroSyntStub,
            **kwargs,
        )

        default_parameters = dict(
            batch_size=32,
            alpha=0.5,
            num_properties=18,
            length_properties=40,
            beam_size=2,
            check_setup=True,
            model="ht-50",
        )
        setup_parameters = {**default_parameters, **setup_parameters}
        setup_parameters["mc_port"] = mc_port
        setup_parameters["verifier"] = verifier

        from ml2.registry import type_from_str

        self.verifier = type_from_str(verifier, bound=Verifier)(
            port=mc_port,
            start_containerized_service=False,
            start_service=False,
        )

        params = {k: json.dumps(v) for k, v in setup_parameters.items()}
        setup_response = self.stub.Setup(SetupRequest(parameters=params))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )
        if not self.assert_identities:
            raise Exception(
                "Tool setup for "
                + self.tool
                + " failed. \n GRPC Server Tool does not match client side"
            )

    def synthesize(self, problem: ToolLTLSynProblem) -> ToolNeuralLTLSynSolution:
        if problem.system_format != "aiger":
            raise NotImplementedError("ML2 can only synthesize AIGER for now.")
        if problem.decomp_specification is None:
            raise NotImplementedError("ML2 can only handle Decomposed LTL specifications for now")
        try:
            return ToolNeuralLTLSynSolution.from_pb2_NeuralLTLSynSolution(
                self.stub.Synthesize(problem.to_pb2_LTLSynProblem())
            )
        except _InactiveRpcError as err:
            return ToolNeuralLTLSynSolution(
                synthesis_solution=ToolLTLSynSolution(
                    status=LTLSynStatus("error"),
                    detailed_status="ERROR:\n" + str(err),
                    tool="NeuroSynt",
                    time=timedelta(0),
                ),
                tool="NeuroSynt",
                time=timedelta(0),
            )

    def synthesize_stream(
        self, problems: Generator[ToolLTLSynProblem, None, None]
    ) -> Generator[ToolNeuralLTLSynSolution, None, None]:
        def _problems(problems):
            for problem in problems:
                if problem.system_format != "aiger":
                    if problem.system_format != "aiger":
                        raise NotImplementedError("ML2 can only synthesize AIGER for now.")
                    if problem.decomp_specification is None:
                        raise NotImplementedError(
                            "ML2 can only handle Decomposed LTL specifications"
                        )
                yield problem.to_pb2_LTLSynProblem()

        for solution in self.stub.SynthesizeStream(_problems(problems)):
            yield ToolNeuralLTLSynSolution.from_pb2_NeuralLTLSynSolution(solution)

    def synthesize_batch(
        self, problems: Generator[ToolLTLSynProblem, None, None]
    ) -> Generator[Tuple[ToolLTLSynProblem, ToolNeuralLTLSynSolution], None, None]:
        def _problems(problems):
            for problem in problems:
                if problem.system_format != "aiger":
                    if problem.system_format != "aiger":
                        raise NotImplementedError("ML2 can only synthesize AIGER for now.")
                    if problem.decomp_specification is None:
                        raise NotImplementedError(
                            "ML2 can only handle Decomposed LTL specifications"
                        )
                yield problem.to_pb2_LTLSynProblem()

        for solution in self.stub.SynthesizeBatch(_problems(problems)):
            yield (
                ToolLTLSynProblem.from_pb2_LTLSynProblem(solution.problem),
                ToolNeuralLTLSynSolution.from_pb2_NeuralLTLSynSolution(solution.solution),
            )
