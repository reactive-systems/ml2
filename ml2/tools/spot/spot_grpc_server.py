"""gRPC Server that checks the satisfiability of an LTL formula and verifies traces using Spot"""

import argparse
from concurrent import futures
import logging
import spot
import time

import grpc

from . import spot_pb2_grpc
from .spot_wrapper import automaton_trace, mc_trace
from ..protos import ltl_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpotServicer(spot_pb2_grpc.SpotServicer):
    def CheckSat(self, request, context):
        start = time.time()
        solution = automaton_trace(request.formula, request.simplify, request.timeout)
        end = time.time()
        logger.info("Checking Satisfiability took %f seconds", end - start)
        return ltl_pb2.LTLSatSolution(
            status=solution["status"].value, trace=solution.get("trace", None)
        )

    def MCTrace(self, request, context):
        start = time.time()
        solution = mc_trace(request.formula, request.trace, request.timeout)
        end = time.time()
        logger.info("Model checking trace took %f seconds", end - start)
        return ltl_pb2.TraceMCSolution(status=solution.value)

    def RandLTL(self, request, context):
        for f in spot.randltl(
            n=request.num_formulas,
            ap=request.aps if request.aps else request.num_aps,
            allow_dups=request.allow_dups,
            output=request.output if request.output else None,
            seed=request.seed,
            simplify=request.simplify,
            tree_size=request.tree_size,
            boolean_priorities=request.boolean_priorities if request.boolean_priorities else None,
            ltl_priorities=request.ltl_priorities if request.ltl_priorities else None,
            sere_priorities=request.sere_priorities if request.sere_priorities else None,
        ):
            yield ltl_pb2.LTLFormula(formula="{0:p}".format(f).replace("xor", "^"))


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    spot_pb2_grpc.add_SpotServicer_to_server(SpotServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spot gRPC server")
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
