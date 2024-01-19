"""gRPC Server that serves SyFCo"""

import argparse
import logging
import os
import random
import subprocess
from concurrent import futures
from datetime import datetime
from typing import Optional, Tuple

import grpc

from ...grpc.syfco import syfco_pb2_grpc
from ...grpc.tools import tools_pb2
from ...ltl.ltl_spec.decomp_ltl_spec import DecompLTLSpec
from ..ltl_tool.tool_ltl_conversion import ToolLTLConversionRequest, ToolLTLConversionResponse

logger = logging.getLogger("SyFCo gRPC Server")


class SyfcoServicer(syfco_pb2_grpc.SyfcoServicer):
    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def Identify(self, request, context):
        return tools_pb2.IdentificationResponse(
            tool="SyFCo",
            functionalities=[
                tools_pb2.FUNCTIONALITY_TLSF_TO_SPEC,
            ],
            version="1.2.1.2",
        )

    def ConvertTLSFToSpec(self, request, context):
        start = datetime.now()
        problem: ToolLTLConversionRequest = (
            ToolLTLConversionRequest.from_pb2_ConvertTLSFToSpecRequest(request)
        )
        result, message = syfco_wrapper_str(problem_str=problem.tlsf_string)
        duration = datetime.now() - start
        print(f"Converting took {duration}")
        if result is not None:
            spec = DecompLTLSpec.from_bosy_str(result)
            return ToolLTLConversionResponse(
                error="", tool="SyFCo", time=duration, specification=spec
            ).to_pb2_ConvertTLSFToSpecResponse()
        else:
            return ToolLTLConversionResponse(
                error=message, tool="SyFCo", time=duration
            ).to_pb2_ConvertTLSFToSpecResponse()


def syfco_wrapper_str(problem_str, temp_dir="/tmp") -> Tuple[Optional[str], str]:
    hash = random.getrandbits(128)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    problem_filepath = os.path.join(temp_dir, "syfco_input_" + str("%032x" % hash) + ".tlsf")
    with open(problem_filepath, "w") as problem_file:
        problem_file.write(problem_str)
    result = syfco_wrapper_file(problem_filepath)
    os.remove(problem_filepath)
    return result


def syfco_wrapper_file(problem_filepath) -> Tuple[Optional[str], str]:
    message = ""
    try:
        out = subprocess.run(
            ["syfco", "-f", "bosy", problem_filepath], capture_output=True, universal_newlines=True
        )
    except subprocess.CalledProcessError as error:
        logger.warning(f"Syfco failed to convert tlsf with error: {error}")
        message = str(error)
    except Exception as error:
        logger.warning(f"Failed to convert tlsf with error: {error}")
        message = str(error)
    else:
        logger.debug("Syfco stdout: %s", out.stdout)
        logger.debug("Syfco stderr: %s", out.stderr)
        if out.stdout == "":
            logger.warning(f"Failed to convert tlsf with error: {out.stderr}")
            message = out.stderr
        else:
            return str(out.stdout).replace("\n", ""), ""
    return None, message


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    syfco_pb2_grpc.add_SyfcoServicer_to_server(SyfcoServicer(), server)
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
