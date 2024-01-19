"""Status of an propositional satisfiability problem"""

from typing import Dict, List

from ..dtypes import CSV, Cat
from ..registry import register_type

PROP_SAT_STATUS_TO_INT = {
    "unsat": 0,
    "sat": 1,
    "error": -1,
    "timeout": -2,
}

INT_TO_PROP_SAT_STATUS = {i: s for s, i in PROP_SAT_STATUS_TO_INT.items()}


@register_type
class PropSatStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["sat", "unsat", "timeout", "error"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"sat": PROP_SAT_STATUS_TO_INT[self._status]}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["sat"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "PropSatStatus":
        return cls(status=INT_TO_PROP_SAT_STATUS[int(fields["sat"])], **kwargs)

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "PropSatStatus":
        return cls(status=token)
