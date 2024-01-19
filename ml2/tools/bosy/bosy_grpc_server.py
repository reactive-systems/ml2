"""gRPC Server that synthesizes LTL specifications using BoSy"""

import argparse
import logging
from concurrent import futures
from datetime import datetime
from typing import Generator

import grpc

from ...grpc import tools_pb2
from ...grpc.bosy import bosy_pb2_grpc
from ...grpc.ltl.ltl_syn_pb2 import LTLSynProblem, LTLSynSolution
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolLTLSynSolution
from .bosy_wrapper import bosy_wrapper

BOSY_PATH = "/root/bosy/.build/release/BoSy"
TEMP_DIR = "/tmp"

logger = logging.getLogger("BoSy gRPC Server")


class BoSyServicer(bosy_pb2_grpc.BosyServicer):
    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def Identify(self, request, context):
        return tools_pb2.IdentificationResponse(
            tool="BoSy",
            functionalities=[tools_pb2.FUNCTIONALITY_LTL_AIGER_SYNTHESIS],
            version="2.0",
        )

    def Synthesize(self, request: LTLSynProblem, context):
        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )

        problem = ToolLTLSynProblem.from_pb2_LTLSynProblem(request)
        assert problem.system_format == "aiger"
        result = bosy_wrapper(
            problem=problem.specification,
            bosy_path=BOSY_PATH,
            temp_dir=TEMP_DIR,
            parameters=problem.parameters,
            timeout=timeout,
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
            circuit=result["circuit"] if "circuit" in result else None,
            mealy_machine=result["mealy_machine"] if "mealy_machine" in result else None,
            realizable=realizable,
            tool="BoSy",
            time=duration,
        ).to_pb2_LTLSynSolution()

    def SynthesizeStream(self, request_iterator, context) -> Generator[LTLSynSolution, None, None]:
        for request in request_iterator:
            yield self.Synthesize(request, context)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    bosy_pb2_grpc.add_BosyServicer_to_server(BoSyServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoSy gRPC server")
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
