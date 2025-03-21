"""gRPC Server for working with ABC, aiger and """

import argparse
import logging
from concurrent import futures
from datetime import datetime

import grpc

from ...aiger.aiger_circuit import AIGERCircuit
from ...grpc import tools_pb2
from ...grpc.abc_aiger import abc_aiger_pb2, abc_aiger_pb2_grpc
from ...grpc.aiger import aiger_pb2
from ...grpc.tools.tools_pb2 import IdentificationResponse
from ..ltl_tool.pb2_converter import TimeConverterPb2
from .abc_wrapper import simplify
from .aiger_wrapper import aag_to_aig, aag_to_dot, aig_to_aag
from .graphviz_wrapper import dot_to_png, dot_to_svg
from .wrapper_helper import RunException

logger = logging.getLogger("ABCAiger gRPC Server")


class ABCAigerServicer(abc_aiger_pb2_grpc.ABCAigerServicer):
    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def Identify(
        self, request: tools_pb2.IdentificationRequest, context
    ) -> IdentificationResponse:
        return tools_pb2.IdentificationResponse(
            tool="ABCAiger",
            functionalities=[],
            version="2.1",
        )

    def ConvertBinaryToAasci(
        self, request: abc_aiger_pb2.ConvertBinaryToAasciRequest, context
    ) -> abc_aiger_pb2.ConvertBinaryToAasciResponse:
        logger.info("ConvertBinaryToAasci called")

        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )
        circuit_bin = request.aig.circuit
        try:
            circuit, out = aig_to_aag(circuit_bin, timeout=timeout)
            duration = datetime.now() - start
            logger.info("ConvertBinaryToAasci took %s seconds", duration)
            return abc_aiger_pb2.ConvertBinaryToAasciResponse(
                aag=aiger_pb2.AigerCircuit(circuit=circuit.to_str()),
                tool="ABCAiger",
                error=out.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )

        except RunException as e:
            duration = datetime.now() - start
            logger.info("ConvertBinaryToAasci failed after %s seconds", duration)
            return abc_aiger_pb2.ConvertBinaryToAasciResponse(
                tool="ABCAiger",
                error=e.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )

    def ConvertAasciToBinary(
        self, request: abc_aiger_pb2.ConvertAasciToBinaryRequest, context
    ) -> abc_aiger_pb2.ConvertAasciToBinaryResponse:
        logger.info("ConvertAasciToBinary called")
        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )
        circuit = AIGERCircuit.from_str(request.aag.circuit)
        try:
            circuit_bin, out = aag_to_aig(circuit, timeout=timeout)
            duration = datetime.now() - start
            logger.info("ConvertAasciToBinary took %s seconds", duration)
            return abc_aiger_pb2.ConvertAasciToBinaryResponse(
                aig=aiger_pb2.AigerBinaryCircuit(circuit=circuit_bin),
                tool="ABCAiger",
                error=out.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )
        except RunException as e:
            duration = datetime.now() - start
            logger.info("ConvertAasciToBinary failed after %s seconds", duration)
            return abc_aiger_pb2.ConvertAasciToBinaryResponse(
                tool="ABCAiger",
                error=e.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )

    def ConvertAigerToDot(
        self, request: abc_aiger_pb2.ConvertAigerToDotRequest, context
    ) -> abc_aiger_pb2.ConvertAigerToDotResponse:
        logger.info("ConvertAigerToDot called")
        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )
        circuit = AIGERCircuit.from_str(request.aag.circuit)
        try:
            dot, out = aag_to_dot(circuit, timeout=timeout)
            duration = datetime.now() - start
            logger.info("ConvertAigerToDot took %s seconds", duration)
            return abc_aiger_pb2.ConvertAigerToDotResponse(
                dot=dot,
                tool="ABCAiger",
                error=out.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )
        except RunException as e:
            duration = datetime.now() - start
            logger.info("ConvertAigerToDot failed after %s seconds", duration)
            return abc_aiger_pb2.ConvertAigerToDotResponse(
                tool="ABCAiger",
                error=e.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )

    def AigerSimplify(
        self, request: abc_aiger_pb2.AigerSimplifyRequest, context
    ) -> abc_aiger_pb2.AigerSimplifyResponse:
        logger.info("AigerSimplify called")
        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )
        circuit = AIGERCircuit.from_str(request.aag.circuit)
        command_sequence = list(request.simplify_commands)
        try:
            circuits, out = simplify(circuit, command_sequence=command_sequence, timeout=timeout)
            duration = datetime.now() - start
            logger.info("AigerSimplify took %s seconds", duration)
            circuits_pb2 = [aiger_pb2.AigerCircuit(circuit=circ.to_str()) for circ in circuits]
            return abc_aiger_pb2.AigerSimplifyResponse(
                aag=circuits_pb2,
                tool="ABCAiger",
                error=out.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )
        except RunException as e:
            duration = datetime.now() - start
            logger.info("AigerSimplify failed after %s seconds", duration)
            return abc_aiger_pb2.AigerSimplifyResponse(
                tool="ABCAiger",
                error=e.serialize(),
                time=TimeConverterPb2(duration).time_to_pb2(),
            )

    def ConvertDotToPng(
        self, request: abc_aiger_pb2.ConvertDotRequest, context
    ) -> abc_aiger_pb2.ConvertDotToPngResponse:
        logger.info("ConvertDotToPng called")
        start = datetime.now()
        dot = request.dot
        png = dot_to_png(dot)
        duration = datetime.now() - start
        logger.info("ConvertDotToPng took %s seconds", duration)
        return abc_aiger_pb2.ConvertDotToPngResponse(
            png=png,
            tool="ABCAiger",
            time=TimeConverterPb2(duration).time_to_pb2(),
        )

    def ConvertDotToSvg(
        self, request: abc_aiger_pb2.ConvertDotRequest, context
    ) -> abc_aiger_pb2.ConvertDotToSvgResponse:
        logger.info("ConvertDotToSvg called")
        start = datetime.now()
        dot = request.dot
        svg = dot_to_svg(dot)
        duration = datetime.now() - start
        logger.info("ConvertDotToSvg took %s seconds", duration)
        return abc_aiger_pb2.ConvertDotToSvgResponse(
            svg=svg,
            tool="ABCAiger",
            time=TimeConverterPb2(duration).time_to_pb2(),
        )


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    abc_aiger_pb2_grpc.add_ABCAigerServicer_to_server(ABCAigerServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABC,  and AIGER gRPC server")
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
