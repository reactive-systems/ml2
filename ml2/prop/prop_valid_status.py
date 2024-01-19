"""Status of a propositional validity problem"""

from typing import Dict, List

from ..dtypes import CSV, Cat
from ..registry import register_type

PROP_VALID_STATUS_TO_INT = {
    "invalid": 0,
    "valid": 1,
    "error": -1,
    "timeout": -2,
}

INT_TO_PROP_VALID_STATUS = {i: s for s, i in PROP_VALID_STATUS_TO_INT.items()}


@register_type
class PropValidStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["valid", "invalid", "timeout", "error"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"valid": PROP_VALID_STATUS_TO_INT[self._status]}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["valid"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "PropValidStatus":
        return cls(status=INT_TO_PROP_VALID_STATUS[int(fields["valid"])])

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "PropValidStatus":
        return cls(status=token)
