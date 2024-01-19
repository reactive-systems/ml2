"""Strix"""

import logging
from datetime import timedelta
from typing import Generator, Union

from grpc._channel import _InactiveRpcError

from ...globals import CONTAINER_REGISTRY
from ...grpc.strix import strix_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_spec import DecompLTLSpec, LTLSpec
from ...ltl.ltl_syn import LTLSynSolution, LTLSynStatus
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from .strix_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRIX_IMAGE_NAME = CONTAINER_REGISTRY + "/strix-grpc-server:latest"


class Strix(GRPCService):
    def __init__(self, image: str = STRIX_IMAGE_NAME, service=serve, **kwargs):
        super().__init__(
            stub=strix_pb2_grpc.StrixStub, image=image, service=service, tool="Strix", **kwargs
        )
        setup_response = self.stub.Setup(SetupRequest(parameters={}))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )

    def synthesize(self, problem: ToolLTLSynProblem) -> ToolLTLSynSolution:
        try:
            return ToolLTLSynSolution.from_pb2_LTLSynSolution(
                self.stub.Synthesize(problem.to_pb2_LTLSynProblem())
            )
        except _InactiveRpcError as err:
            return ToolLTLSynSolution(
                status=LTLSynStatus("error"),
                detailed_status="ERROR:\n" + str(err),
                tool="Strix",
                time=timedelta(0),
            )

    def synthesize_spec(
        self,
        spec: Union[LTLSpec, DecompLTLSpec],
        minimize_aiger=False,
        minimize_mealy=False,
        timeout=None,
    ) -> LTLSynSolution:
        minimize_mealy: dict[str, str] = (
            {"--minimize": "both"} if minimize_mealy else {"--minimize": "none"}
        )
        minimize_aiger = {"--compression": "more"} if minimize_aiger else {"--compression": "none"}
        timeout = {"timeout": timeout} if timeout else {}
        problem = ToolLTLSynProblem(
            parameters={
                **minimize_aiger,
                **minimize_mealy,
                **timeout,
            },
            specification=spec,
        )
        solution = self.synthesize(problem=problem)
        return LTLSynSolution(
            status=solution.status,
            circuit=solution.circuit,
            time=solution.time_seconds,
            tool=solution.tool,
        )

    def synthesize_stream(
        self,
        problems: Generator[ToolLTLSynProblem, None, None],
    ) -> Generator[ToolLTLSynSolution, None, None]:
        def _problems(problems):
            for problem in problems:
                if problem.system_format != "aiger":
                    raise NotImplementedError(
                        "Strix can only synthesize AIGER for now. New Strix version would support HOA too."
                    )
                yield problem.to_pb2_LTLSynProblem()

        for solution in self.stub.SynthesizeStream(_problems(problems)):
            yield ToolLTLSynSolution.from_pb2_LTLSynSolution(solution)
