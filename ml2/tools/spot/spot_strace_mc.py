"""Spot symbolic trace model checker"""

from ...ltl import LTLFormula
from ...registry import register_type
from ...trace import SymbolicTrace, TraceMCStatus
from ...verifier import Verifier
from .spot import Spot


@register_type
class SpotSTraceMC(Spot, Verifier):
    def verify(
        self, problem: LTLFormula, solution: SymbolicTrace, timeout: int = 600
    ) -> TraceMCStatus:
        return self.mc_trace(
            formula=problem.to_str(notation="infix"),
            trace=solution.to_str(notation="infix", spot=True),
            timeout=timeout,
        )
