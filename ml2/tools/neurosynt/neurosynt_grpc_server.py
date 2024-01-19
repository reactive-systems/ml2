"""gRPC Server that synthesizes LTL specifications using NeuroSynt"""

import argparse
import json
import logging
from concurrent import futures
from datetime import datetime
from typing import Generator

import grpc

from ...grpc.ltl.ltl_syn_pb2 import LTLSynProblem, NeuralLTLSynSolution
from ...grpc.neurosynt import neurosynt_pb2_grpc
from ...grpc.tools import tools_pb2
from ...ltl.ltl_spec.decomp_ltl_spec import DecompLTLSpec
from ..ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem, ToolNeuralLTLSynSolution
from .pipeline_wrapper import PipelineWrapper

logger = logging.getLogger("NeuroSynt gRPC Server")


class NeuroSyntServicer(neurosynt_pb2_grpc.NeuroSyntServicer):
    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        parameters = {k: json.loads(v) for k, v in request.parameters.items()}

        logger.info(str(parameters))
        self.pipeline = PipelineWrapper(
            beam_size=int(parameters["beam_size"]),
            model=parameters["model"],
            mc_port=int(parameters["mc_port"]),
            verifier=parameters["verifier"],
            batch_size=int(parameters["batch_size"]),
            alpha=float(parameters["alpha"]),
            num_properties=int(parameters["num_properties"]),
            length_properties=int(parameters["length_properties"]),
        )
        if parameters["check_setup"]:
            try:
                startup_spec = DecompLTLSpec.from_dict(
                    {"guarantees": ["G (i -> F o)"], "inputs": ["i"], "outputs": ["o"]}
                )
                solution = self.pipeline.eval_sample(startup_spec)
                if solution[0].status.token() != "realizable":
                    return tools_pb2.SetupResponse(
                        success=False,
                        error="SynStatus realizable expected but got"
                        + solution[0].status.token()
                        + "."
                        + (
                            " Circuit: " + solution[0].circuit.to_str()
                            if solution[0].circuit is not None
                            else ""
                        ),
                    )
                if solution[1].status.token() != "satisfied":
                    return tools_pb2.SetupResponse(
                        success=False,
                        error="MCStatus satisfied expected but got"
                        + solution[1].status.token()
                        + "."
                        + (
                            " Circuit: " + solution[0].circuit.to_str()
                            if solution[0].circuit is not None
                            else ""
                        ),
                    )
                return tools_pb2.SetupResponse(success=True, error="")
            except Exception as e:
                return tools_pb2.SetupResponse(success=False, error=str(e))
        else:
            return tools_pb2.SetupResponse(success=True, error="")

    def Identify(self, request, context):
        return tools_pb2.IdentificationResponse(
            tool="NeuroSynt",
            functionalities=[tools_pb2.FUNCTIONALITY_NEURAL_LTL_AIGER_SYNTHESIS],
            version="2.1",
        )

    def Synthesize(self, request: LTLSynProblem, context) -> NeuralLTLSynSolution:
        start = datetime.now()
        problem = ToolLTLSynProblem.from_pb2_LTLSynProblem(request)
        assert problem.decomp_specification is not None
        assert problem.system_format == "aiger"
        allow_unsound = (
            "allow_unsound" in problem.parameters and problem.parameters["allow_unsound"]
        )
        smallest_result = (
            "smallest_result" not in problem.parameters or problem.parameters["smallest_result"]
        )
        solution, mc_solution = self.pipeline.eval_sample(
            problem.decomp_specification,
            allow_unsound=allow_unsound,
            smallest_result=smallest_result,
        )
        duration = datetime.now() - start
        print(f"Synthesizing took {duration}")
        return ToolNeuralLTLSynSolution(
            synthesis_solution=solution,
            tool="NeuroSynt",
            time=duration,
            model_checking_solution=mc_solution,
        ).to_pb2_NeuralLTLSynSolution()

    def SynthesizeStream(
        self, request_iterator, context
    ) -> Generator[NeuralLTLSynSolution, None, None]:
        for request in request_iterator:
            yield self.Synthesize(request, context)

    def SynthesizeBatch(self, request_iterator, context):
        raise NotImplementedError


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    neurosynt_pb2_grpc.add_NeuroSyntServicer_to_server(NeuroSyntServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroSynt gRPC server")
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
