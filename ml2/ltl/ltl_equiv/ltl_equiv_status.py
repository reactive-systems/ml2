"""Status of an LTL equivalence checking problem"""

from typing import Dict, List

from ...dtypes import CSV, Cat
from ...verifier import EquivStatus

STATUS_TO_INT = {
    "equivalent": 1,
    "inequivalent": 0,
    "error": -1,
    "timeout": -2,
}

INT_TO_STATUS = {v: k for k, v in STATUS_TO_INT.items()}


class LTLEquivStatus(EquivStatus, Cat, CSV):
    status_to_int = STATUS_TO_INT
    int_to_status = INT_TO_STATUS

    def __init__(self, status: str) -> None:
        if status not in ["equivalent", "inequivalent", "timeout", "error"]:
            raise Exception(f"Invalid status {status}")
        self._status = status

    @property
    def equiv(self) -> bool:
        if self._status not in ["equivalent", "inequivalent"]:
            raise Exception(f"Cannot determine equivalence of status {self._status}")
        return self._status == "equivalent"

    @property
    def status(self) -> str:
        return self._status

    def token(self, **kwargs) -> str:
        return self._status

    def to_int(self) -> int:
        return self.status_to_int[self._status]

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"equiv": str(self.to_int())}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["equiv"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLEquivStatus":
        return cls.from_int(int(fields["equiv"]))

    @classmethod
    def from_int(cls, i: int) -> "LTLEquivStatus":
        return cls(status=cls.int_to_status[i])

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "LTLEquivStatus":
        return cls(status=token)
