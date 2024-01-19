"""gRPC Server that checks the satisfiability of an LTL formula and verifies traces using Spot"""

import argparse
import json
import logging
import time
from concurrent import futures
from datetime import datetime
from multiprocessing import Manager, Process, set_start_method
from typing import Dict, Generator, List, Set

import grpc

from ...aiger import AIGERCircuit
from ...grpc import tools_pb2
from ...grpc.aiger import aiger_pb2
from ...grpc.ltl import (
    ltl_equiv_pb2,
    ltl_mc_pb2,
    ltl_pb2,
    ltl_sat_pb2,
    ltl_syn_pb2,
    ltl_trace_mc_pb2,
)
from ...grpc.mealy import mealy_pb2
from ...grpc.spot import spot_pb2_grpc
from ...ltl.ltl_mc import LTLMCStatus
from ...trace import SymbolicTrace, TraceMCStatus
from ..ltl_tool import ToolLTLMCProblem, ToolLTLMCSolution, ToolLTLSynProblem, ToolLTLSynSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpotServicer(spot_pb2_grpc.SpotServicer):
    def __init__(self):
        # set to spawn to avoid problems with creating new processes in ThreadPoolExecutor (see: https://stackoverflow.com/questions/67273533/processes-in-python-3-9-exiting-with-code-1-when-its-created-inside-a-threadpoo)
        set_start_method("spawn")
        self.manager = Manager()

    def Synthesize(
        self, request: ltl_syn_pb2.LTLSynProblem, context
    ) -> ltl_syn_pb2.LTLSynSolution:
        from .spot_wrapper import ltlsynt

        start = datetime.now()
        timeout = (
            float(request.parameters.pop("timeout")) if "timeout" in request.parameters else None
        )

        problem = ToolLTLSynProblem.from_pb2_LTLSynProblem(request)
        result = ltlsynt(
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
            mealy_machine=result["mealy_machine"] if "mealy_machine" in result else None,
            realizable=realizable,
            tool="Spot",
            time=duration,
        ).to_pb2_LTLSynSolution()

    def ModelCheck(self, request: ltl_mc_pb2.LTLMCProblem, context) -> ltl_mc_pb2.LTLMCSolution:
        from .spot_wrapper import model_check_system

        start = datetime.now()
        timeout = float(request.parameters["timeout"]) if "timeout" in request.parameters else None
        problem = ToolLTLMCProblem.from_pb2_LTLMCProblem(request)

        # TODO multiprocessing introduces much overhead and model checking takes much longer than without multiprocessing. But it seems necessary to implement a timeout that is able to terminate the process.
        result = self.manager.dict()
        process = Process(
            target=model_check_system,
            args=(
                problem.specification.to_str(notation="infix"),
                problem.circuit.to_str() if problem.circuit is not None else None,
                problem.mealy_machine.to_hoa(realizable=problem.realizable)
                if problem.mealy_machine is not None
                else None,
                problem.realizable,
                result,
            ),
        )
        process.start()
        process.join(timeout=timeout)
        duration = datetime.now() - start

        logger.info(
            "Multiprocessing and model checking AIGER circuit took %s seconds", str(duration)
        )

        if process.exitcode == 0:
            detailed_status = result["status"].token().upper()
            try:
                counterexample = (
                    SymbolicTrace.from_str(result["counterexample"], spot=True)
                    if result["counterexample"] is not None
                    else None
                )
            except Exception as error:
                counterexample = None
                detailed_status = detailed_status + " Parsing Counterexample failed: " + str(error)
            return ToolLTLMCSolution(
                status=result["status"],
                detailed_status=detailed_status,
                tool="Spot",
                time=result["time"],
                counterexample=counterexample,
            ).to_pb2_LTLMCSolution()
        elif process.is_alive():
            process.terminate()
            return ToolLTLMCSolution(
                status=LTLMCStatus("timeout"),
                detailed_status="TIMEOUT",
                tool="Spot",
            ).to_pb2_LTLMCSolution()
        else:
            return ToolLTLMCSolution(
                status=LTLMCStatus("error"),
                detailed_status="ERROR",
                tool="Spot",
            ).to_pb2_LTLMCSolution()

    def Setup(self, request: tools_pb2.SetupRequest, context) -> tools_pb2.SetupResponse:
        logger.info(str(request.parameters))
        return tools_pb2.SetupResponse(success=True, error="")

    def ModelCheckStream(
        self, request_iterator, context
    ) -> Generator[ltl_mc_pb2.LTLMCSolution, None, None]:
        for request in request_iterator:
            yield self.ModelCheck(request, context)

    def Identify(self, request, context):
        return tools_pb2.IdentificationResponse(
            tool="Spot",
            functionalities=[
                tools_pb2.FUNCTIONALITY_LTL_AIGER_MODELCHECKING,
                tools_pb2.FUNCTIONALITY_LTL_MEALY_MODELCHECKING,
                tools_pb2.FUNCTIONALITY_LTL_EQUIVALENCE,
                tools_pb2.FUNCTIONALITY_LTL_TRACE_MODELCHECKING,
                tools_pb2.FUNCTIONALITY_RANDLTL,
                tools_pb2.FUNCTIONALITY_AIGER_TO_MEALY,
                tools_pb2.FUNCTIONALITY_MEALY_TO_AIGER,
                tools_pb2.FUNCTIONALITY_LTL_AIGER_SYNTHESIS,
            ],
            version="2.11.6",
        )

    def CheckEquiv(self, request, context):
        from .spot_wrapper import check_equiv

        start = time.time()
        if request.HasField("timeout"):
            result = self.manager.dict()
            process = Process(
                target=check_equiv, args=(request.formula1, request.formula2, result)
            )
            process.start()
            process.join(timeout=request.timeout)
            process.terminate()
            end = time.time()
            logger.info("Multiprocessing and checking equivalence took %f seconds", end - start)
            if process.exitcode == 0:
                return ltl_equiv_pb2.LTLEquivSolution(
                    status=result["status"],
                    time=result["time"],
                )
            elif process.exitcode is None:
                return ltl_equiv_pb2.LTLEquivSolution(status="timeout")
            else:
                return ltl_equiv_pb2.LTLEquivSolution(status="error")
        else:
            result = {}
            check_equiv(request.formula1, request.formula2, result)
            end = time.time()
            logger.info("Checking equivalence took %f seconds", end - start)
            return ltl_equiv_pb2.LTLEquivSolution(status=result["status"], time=result["time"])

    def CheckSat(self, request, context):
        from .spot_wrapper import check_sat

        solution = check_sat(request.formula, request.simplify, int(request.timeout))
        return ltl_sat_pb2.LTLSatSolution(
            status=solution["status"].value, trace=solution.get("trace", None)
        )

    def MCTrace(self, request, context):
        from .spot_wrapper import mc_trace

        start = datetime.now()
        result = self.manager.dict()
        process = Process(
            target=mc_trace,
            args=(request.formula, request.trace, result),
        )
        # mc_trace(request.formula, request.trace, result)
        process.start()
        process.join(timeout=int(request.timeout) if request.timeout.isdigit() else None)
        # process.join(timeout=None)
        duration = datetime.now() - start

        logger.info(
            "Multiprocessing and model checking AIGER circuit took %s seconds", str(duration)
        )
        if process.exitcode == 0:
            return ltl_trace_mc_pb2.LTLTraceMCSolution(status=result["status"].token().upper())
        elif process.is_alive():
            process.terminate()
            return ltl_trace_mc_pb2.LTLTraceMCSolution(
                status=TraceMCStatus("timeout").token().upper()
            )
        else:
            return ltl_trace_mc_pb2.LTLTraceMCSolution(
                status=TraceMCStatus("error").token().upper()
            )

    def RandLTL(self, request, context):
        import spot

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

    def aag2mealy(self, request, context):
        import spot

        start = time.time()
        aiger = spot.aiger_circuit(request.circuit)
        mealy = aiger.as_automaton()
        end = time.time()
        logger.info("Converting the circuit to a mealy machine took %f seconds", end - start)
        return mealy_pb2.MealyMachine(machine=mealy.to_str())

    def mealy2aag(self, request, context):
        import spot

        start = time.time()
        mealy = spot.automaton(request.mealy)
        aiger = spot.mealy_machine_to_aig(mealy, "isop")
        end = time.time()
        logger.info("Converting the mealy machine into a circuit took %f seconds", end - start)
        return aiger_pb2.AigerCircuit(circuit=aiger.to_str())

    def extractTransitions(self, request, context):
        import spot

        hoa = request.mealy
        aut = spot.automaton(hoa.replace('""', '"'))
        bdict = aut.get_dict()
        init = aut.get_init_state_number()

        assert type(aut.get_init_state_number()) == int
        assert aut.prop_universal() and aut.is_existential()

        edges = []

        def bf_search(
            queue: List[int],
            aut,
            bdict,
            visited: Set[int] = set(),
            rename: Dict = {},
        ):
            import spot

            s = queue.pop(0)
            visited.add(s)
            if s not in rename.keys():
                rename[s] = len(rename)
            for edge in aut.out(s):
                if edge.dst not in rename.keys():
                    rename[edge.dst] = len(rename)
                edges.append(
                    {
                        "src": rename[edge.src],
                        "cond": spot.bdd_format_formula(bdict, edge.cond),
                        "dst": rename[edge.dst],
                    }
                )
                if edge.dst not in visited:
                    queue.append(edge.dst)
            if queue:
                bf_search(queue, aut, bdict, visited, rename)

        bf_search([init], aut, bdict, set(), dict())

        return mealy_pb2.MealyTransitions(transitions=json.dumps(edges))


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
