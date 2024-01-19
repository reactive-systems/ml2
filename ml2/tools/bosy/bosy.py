"""BoSy"""

import logging
from datetime import timedelta
from typing import Generator, Union

from grpc._channel import _InactiveRpcError

from ...globals import CONTAINER_REGISTRY
from ...grpc.bosy import bosy_pb2_grpc
from ...grpc.tools.tools_pb2 import SetupRequest
from ...ltl.ltl_spec import DecompLTLSpec, LTLSpec
from ...ltl.ltl_syn import LTLSynSolution, LTLSynStatus
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from .bosy_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOSY_IMAGE_NAME = CONTAINER_REGISTRY + "/bosy-grpc-server:latest"


class BoSy(GRPCService):
    def __init__(self, image: str = BOSY_IMAGE_NAME, cpu_count: int = 1, service=serve, **kwargs):
        super().__init__(
            stub=bosy_pb2_grpc.BosyStub,
            image=image,
            cpu_count=cpu_count,
            tool="BoSy",
            service=service,
            **kwargs
        )
        setup_response = self.stub.Setup(SetupRequest(parameters={}))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )
        # logger.info("Compiling BoSy ...")
        # spec = DecompLTLSpec.from_dict(
        #     {"guarantees": ["G (i -> F o)"], "inputs": ["i"], "outputs": ["o"]}
        # )
        # self.synthesize(spec)
        # logger.info("Compiled Bosy")

    def synthesize(self, problem: ToolLTLSynProblem) -> ToolLTLSynSolution:
        if problem.system_format != "aiger":
            raise NotImplementedError("BoSy can only synthesize AIGER.")
        try:
            return ToolLTLSynSolution.from_pb2_LTLSynSolution(
                self.stub.Synthesize(problem.to_pb2_LTLSynProblem())
            )
        except _InactiveRpcError as err:
            return ToolLTLSynSolution(
                status=LTLSynStatus("error"),
                detailed_status="ERROR:\n" + str(err),
                tool="BoSy",
                time=timedelta(0),
            )

    def synthesize_spec(
        self, spec: Union[LTLSpec, DecompLTLSpec], optimize: bool = False, timeout=None
    ) -> LTLSynSolution:
        timeout = {"timeout": timeout} if timeout else {}
        optimize: dict[str, str] = {"--optimize": ""} if optimize else {}
        problem = ToolLTLSynProblem(
            parameters={**optimize, **timeout},
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
                    raise NotImplementedError("BoSy can only synthesize AIGER.")
                yield problem.to_pb2_LTLSynProblem()

        for solution in self.stub.SynthesizeStream(_problems(problems)):
            yield ToolLTLSynSolution.from_pb2_LTLSynSolution(solution)
