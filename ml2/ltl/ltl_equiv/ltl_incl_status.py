"""Status of an LTL equivalence checking problem"""

from typing import Dict, List

from ...dtypes import CSV, Cat
from .ltl_equiv_status import LTLEquivStatus

STATUS_TO_INT = {
    "equivalent": 1,
    "incomparable": 0,
    "only_left_in_right": 2,
    "only_right_in_left": 3,
    "error": -1,
    "timeout": -2,
}

INT_TO_STATUS = {v: k for k, v in STATUS_TO_INT.items()}


class LTLInclStatus(LTLEquivStatus, Cat, CSV):
    status_to_int = STATUS_TO_INT
    int_to_status = INT_TO_STATUS

    @property
    def valid(self) -> bool:
        return self._status in [
            "equivalent",
            "incomparable",
            "only_left_in_right",
            "only_right_in_left",
        ]

    @property
    def left_in_right(self) -> bool:
        if not self.valid:
            raise Exception(f"Cannot determine language inclusion of status {self._status}")
        return self._status == "only_left_in_right" or self.equiv

    @property
    def right_in_left(self) -> bool:
        if not self.valid:
            raise Exception(f"Cannot determine language inclusion of status {self._status}")
        return self._status == "only_right_in_left" or self.equiv

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"incl": str(self.to_int())}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["incl"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLInclStatus":
        return cls.from_int(int(fields["incl"]))

    @classmethod
    def from_int(cls, i: int) -> "LTLInclStatus":
        return cls(status=cls.int_to_status[i])

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "LTLInclStatus":
        return cls(status=token)
