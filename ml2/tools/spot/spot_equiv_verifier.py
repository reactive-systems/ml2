"""Spot symbolic trace model checker"""

from ...dtypes import String
from ...ltl.ltl_equiv import LTLEquivStatus
from ...verifier import EquivVerifier
from .spot import Spot


class SpotEquivVerifier(Spot, EquivVerifier):
    def verify_equiv(self, x: String, y: String, **kwargs) -> LTLEquivStatus:
        status, time = self.check_equiv(formula1=x.to_str(), formula2=y.to_str(), timeout=10)
        return LTLEquivStatus(status=status)
