"""Spot AIGER model checker"""

from typing import Dict, Optional

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq
from ...ltl.ltl_mc import LTLMCSolution
from ...ltl.ltl_spec.decomp_ltl_spec import DecompLTLSpec
from ...ltl.ltl_syn.ltl_real_status import LTLRealStatus
from ...registry import register_type
from ...verifier import Verifier
from ..ltl_tool.tool_ltl_mc_problem import ToolLTLMCProblem
from .spot import Spot


@register_type
class SpotAIGERMC(Spot, Verifier):
    def verify(
        self,
        formula: DecompLTLSpec,
        solution: CatSeq[LTLRealStatus, AIGERCircuit],
        parameters: Optional[Dict[str, str]] = None,
    ) -> LTLMCSolution:
        if parameters is None:
            parameters = {}
        if "timeout" not in parameters:
            parameters["timeout"] = 120
        return self.model_check(
            problem=ToolLTLMCProblem.from_aiger_verification_pair(
                formula=formula, solution=solution, parameters=parameters
            )
        ).to_LTLMCSolution()
