"""Status of an assignment check"""

from typing import Dict, List

from ..dtypes import CSV, Cat
from ..registry import register_type

ASSIGN_CHECK_STATUS_TO_INT = {
    "satisfying": 1,
    "unsatisfying": 0,
    "error": -1,
    "timeout": -2,
}

INT_TO_ASSIGN_CHECK_STATUS = {i: s for s, i in ASSIGN_CHECK_STATUS_TO_INT.items()}


@register_type
class AssignmentCheckStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["satisfying", "unsatisfying", "timeout", "error"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"satisfying": ASSIGN_CHECK_STATUS_TO_INT[self._status]}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["satisfying"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "AssignmentCheckStatus":
        return cls(status=INT_TO_ASSIGN_CHECK_STATUS[int(fields["satisfying"])], **kwargs)

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "AssignmentCheckStatus":
        return cls(status=token)
