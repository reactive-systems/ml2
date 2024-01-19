"""Status of an LTL satisfiability problem"""

from typing import Dict, List

from ...dtypes.cat import Cat
from ...dtypes.csv_dtype import CSV
from ...registry import register_type

LTL_SAT_STATUS_TO_INT = {
    "satisfiable": 1,
    "unsatisfiable": 0,
    "error": -1,
    "timeout": -2,
}

INT_TO_LTL_SAT_STATUS = {v: k for k, v in LTL_SAT_STATUS_TO_INT.items()}


@register_type
class LTLSatStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["satisfiable", "unsatisfiable", "timeout", "error"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"sat": LTL_SAT_STATUS_TO_INT[self._status]}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["sat"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLSatStatus":
        return cls(status=INT_TO_LTL_SAT_STATUS[int(fields["sat"])])

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "LTLSatStatus":
        return cls(status=token)
