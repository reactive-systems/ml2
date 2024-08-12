"""Status of resolution proof checking problem"""

from typing import Dict, List, Optional

from ...dtypes import CSV, Cat

RES_PROOF_CHECK_STATUS_TO_INT = {
    "resolved": 0,
    "failed": 1,
    "error": -1,
    "timeout": -2,
}

INT_TO_RES_PROOF_CHECK_STATUS = {i: s for s, i in RES_PROOF_CHECK_STATUS_TO_INT.items()}


class ResProofCheckStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["resolved", "failed", "timeout", "error"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"status": RES_PROOF_CHECK_STATUS_TO_INT[self._status]}

    @property
    def validation_success(self) -> Optional[bool]:
        return self._status == "resolved" if self._status is not None else None

    @property
    def validation_status(self) -> Optional[str]:
        return self._status if self._status is not None else None

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["status"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "ResProofCheckStatus":
        return cls(status=INT_TO_RES_PROOF_CHECK_STATUS[int(fields["status"])])

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "ResProofCheckStatus":
        return cls(status=token)
