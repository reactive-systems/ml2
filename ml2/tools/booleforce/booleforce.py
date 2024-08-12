"""BooleForce"""

import logging
from copy import deepcopy
from typing import Dict, Optional

from ...globals import CONTAINER_REGISTRY
from ...grpc.booleforce import booleforce_pb2_grpc
from ...grpc.prop import prop_pb2
from ...prop import AssignmentCheckStatus, PropSatStatus
from ...prop.cnf import (
    Clause,
    CNFAssignment,
    CNFFormula,
    CNFSatSearchSolution,
    ResProof,
    ResProofCheckStatus,
)
from ...verifier import Verifier
from ..grpc_service import GRPCService
from .booleforce_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_NAME = CONTAINER_REGISTRY + "/booleforce-grpc-server:latest"


class BooleForce(GRPCService):
    def __init__(
        self,
        cpu_count: int = 1,
        image=IMAGE_NAME,
        service=serve,
        tool: str = "BooleForce",
        **kwargs,
    ):
        super().__init__(
            cpu_count=cpu_count,
            image=image,
            service=service,
            stub=booleforce_pb2_grpc.BooleForceStub,
            tool=tool,
            **kwargs,
        )

    def check_sat(self, formula: CNFFormula, timeout: float = None) -> CNFSatSearchSolution:
        pb_problem = prop_pb2.CNFSatProblem(formula=formula.pb, timeout=timeout)
        pb_solution = self.stub.CheckSat(pb_problem)
        # cast assignment to python list (from google.protobuf.pyext._message.RepeatedScalarContainer) important for using ray
        solution = CNFSatSearchSolution(
            status=PropSatStatus(pb_solution.status),
            assignment=list(pb_solution.assignment),
            res_proof=ResProof.from_pb(pb_solution.res_proof),
            solver="BooleForce",
            time=pb_solution.time,
        )
        return solution

    def check_assignment(
        self, formula: CNFFormula, assignment: CNFAssignment, timeout: float = None
    ) -> AssignmentCheckStatus:
        conjunct_formula = deepcopy(formula)
        for lit in assignment.assignment:
            conjunct_formula.add_clause(Clause.from_list([lit]))
        sat_sol = self.check_sat(conjunct_formula, timeout)
        sat_status = sat_sol.status
        if sat_status == PropSatStatus("sat"):
            return AssignmentCheckStatus("satisfying")
        elif sat_status == PropSatStatus("unsat"):
            return AssignmentCheckStatus("unsatisfying")
        elif sat_status == PropSatStatus("timeout"):
            return AssignmentCheckStatus("timeout")
        elif sat_status == PropSatStatus("error"):
            return AssignmentCheckStatus("error")
        else:
            raise ValueError(f"Unknown SAT status: {sat_status}")

    def trace_check(self, proof: ResProof, timeout: float = None) -> ResProofCheckStatus:
        pb_problem = prop_pb2.ResProofCheckProblem(proof=proof.pb, timeout=timeout)
        pb_solution = self.stub.TraceCheck(pb_problem)
        return ResProofCheckStatus(pb_solution.status)

    def binarize_res_proof(self, proof: ResProof) -> ResProof:
        pb_res_proof = prop_pb2.ResProof(proof=proof.pb)
        pb_bin_res_proof = self.stub.BinarizeResProof(pb_res_proof)
        return ResProof.from_pb(pb_bin_res_proof.proof)


class TraceCheckVerifier(BooleForce, Verifier):
    def verify(
        self,
        problem: CNFFormula,
        solution: ResProof,
        parameters: Optional[Dict[str, str]] = None,
    ) -> ResProofCheckStatus:
        if parameters is None:
            parameters = {}
        result = self.trace_check(proof=solution)
        return result
