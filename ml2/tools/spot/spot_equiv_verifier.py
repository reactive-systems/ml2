"""Spot symbolic trace model checker"""

from ...ltl.ltl_equiv import LTLEquivStatus
from ...ltl.ltl_formula import LTLFormula
from ...verifier import EquivVerifier
from .spot import Spot


class SpotEquivVerifier(Spot, EquivVerifier):
    def verify_equiv(self, x: LTLFormula, y: LTLFormula, **kwargs) -> LTLEquivStatus:
        return self.check_equiv(f=x, g=y, timeout=10)
