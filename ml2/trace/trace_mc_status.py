"""Status of an trace model checking problem"""

from typing import Dict, List, Optional

from ..dtypes import CSV, Cat

TRACE_MC_STATUS_TO_INT = {
    "satisfied": 1,
    "violated": 0,
    "error": -1,
    "timeout": -2,
    "invalid": -3,
    "nonsuccess": -4,
}

INT_TO_LTL_MC_STATUS = {v: k for k, v in TRACE_MC_STATUS_TO_INT.items()}


class TraceMCStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["satisfied", "violated", "error", "timeout", "invalid", "nonsuccess"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    @classmethod
    def from_int(cls, i: int) -> "TraceMCStatus":
        return cls(status=INT_TO_LTL_MC_STATUS[i])

    def to_int(self) -> int:
        return TRACE_MC_STATUS_TO_INT[self._status]

    @property
    def satisfied(self):
        return self._status == "satisfied"

    @property
    def validation_success(self) -> Optional[bool]:
        """Return true if validiation was succesfull"""
        return self._status == "satisfied"

    @property
    def validation_status(self) -> Optional[str]:
        """Return more detailed status of validation"""
        return self._status

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"satisfied": str(self.to_int())}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["satisfied"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "TraceMCStatus":
        # TODO change realizable to status in datasets
        return cls.from_int(int(fields["satisfied"]))

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "TraceMCStatus":
        return cls(status=token)
