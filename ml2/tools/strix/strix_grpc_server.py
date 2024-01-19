"""gRPC Server that synthesizes LTL specifications using Strix"""

import argparse
import logging
from concurrent import futures
from datetime import datetime
from typing import Generator

import grpc

from ...aiger.aiger_circuit import AIGERCircuit
from ...grpc import tools_pb2
from ...grpc.ltl.ltl_syn_pb2 import LTLSynProblem, LTLSynSolution
from ...grpc.strix import strix_pb2_grpc
from ...grpc.tools.tools_pb2 import IdentificationResponse
from ...mealy.mealy_machine import MealyMachine
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from .strix_wrapper import strix_wrapper_str

logger = logging.getLogger("Strix gRPC Server")


class StrixServicer(strix_pb2_grpc.StrixServicer):
    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def Identify(
        self, request: tools_pb2.IdentificationRequest, context
    ) -> IdentificationResponse:
        return tools_pb2.IdentificationResponse(
            tool="Strix",
            functionalities=[
                tools_pb2.FUNCTIONALITY_LTL_AIGER_SYNTHESIS,
                tools_pb2.FUNCTIONALITY_LTL_MEALY_SYNTHESIS,
            ],
            version="2.1",
        )

    def Synthesize(self, request: LTLSynProblem, context) -> LTLSynSolution:
        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )

        problem = ToolLTLSynProblem.from_pb2_LTLSynProblem(request)
        result = strix_wrapper_str(
            formula_str=problem.specification.to_str(),
            ins_str=problem.specification.input_str,
            outs_str=problem.specification.output_str,
            parameters=problem.parameters,
            timeout=timeout,
            system_format=problem.system_format,
        )
        duration = datetime.now() - start
        print(f"Synthesizing took {duration}")
        try:
            realizable = result["status"].realizable
        except Exception:
            realizable = None
        return ToolLTLSynSolution(
            status=result["status"],
            detailed_status=result["status"].token().upper()
            + (":\n" + result["message"] if "message" in result else ""),
            circuit=AIGERCircuit.from_str(result["circuit"]) if "circuit" in result else None,
            mealy_machine=MealyMachine.from_hoa(result["mealy_machine"])
            if "mealy_machine" in result
            else None,
            realizable=realizable,
            tool="Strix",
            time=duration,
        ).to_pb2_LTLSynSolution()

    def SynthesizeStream(self, request_iterator, context) -> Generator[LTLSynSolution, None, None]:
        for request in request_iterator:
            yield self.Synthesize(request, context)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    strix_pb2_grpc.add_StrixServicer_to_server(StrixServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strix gRPC server")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=50051,
        metavar="port number",
        help=("port on which server accepts RPCs"),
    )
    args = parser.parse_args()
    serve(args.port)
