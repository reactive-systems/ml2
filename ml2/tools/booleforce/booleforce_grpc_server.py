"""gRPC Server that checks the satisfiability of an CNF formula using BooleForce"""

import argparse
import logging
import os
import time
from concurrent import futures

import grpc

from ...grpc.booleforce import booleforce_pb2_grpc
from ...grpc.prop import prop_pb2
from ...prop.cnf import CNFFormula, ResProof
from .booleforce_wrapper import (
    booleforce_sat_wrapper,
    booleforce_trace_check_wrapper,
    trace_check_binarize,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = "/tmp"


class BooleForceServicer(booleforce_pb2_grpc.BooleForceServicer):
    def CheckSat(self, request, context):
        start = time.time()
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        filepath = os.path.join(TEMP_DIR, "problem.cnf")
        CNFFormula.from_pb(request.formula).to_dimacs_file(filepath)
        solution = booleforce_sat_wrapper(
            filepath,
            timeout=request.timeout if request.timeout > 0 else None,
        )
        end = time.time()
        print(f"Checking satisfiability took {end - start} seconds")
        return prop_pb2.CNFSatSolution(
            status=solution["status"].token(),
            assignment=solution.get("assignment", None),
            res_proof=solution.get("res_proof", None),
            time=end - start,
        )

    def TraceCheck(self, request, context):
        start = time.time()
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        filepath = os.path.join(TEMP_DIR, "trace")
        ResProof.from_pb(request.proof).to_tracecheck_file(filepath)
        solution = booleforce_trace_check_wrapper(
            filepath,
            timeout=request.timeout if request.timeout > 0 else None,
        )
        end = time.time()
        print(f"Checking trace took {end - start} seconds")
        return prop_pb2.ResProofCheckSolution(status=solution["status"].token())

    def BinarizeResProof(self, request, context):
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        proof_filepath = os.path.join(TEMP_DIR, "proof")
        binarized_proof_filepath = os.path.join(TEMP_DIR, "binarized_proof")
        ResProof.from_pb(request.proof).to_tracecheck_file(proof_filepath)
        status = trace_check_binarize(proof_filepath, binarized_proof_filepath)
        res_proof = ResProof.from_tracecheck_file(binarized_proof_filepath)
        return prop_pb2.ResProof(proof=res_proof.pb)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    booleforce_pb2_grpc.add_BooleForceServicer_to_server(BooleForceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BooleForce gRPC server")
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
