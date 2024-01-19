"""Status of an LTL model checking problem"""

from typing import Dict, List

from ...dtypes.cat import Cat
from ...dtypes.csv_dtype import CSV
from ...grpc.ltl import ltl_mc_pb2

LTL_MC_STATUS_TO_INT = {
    "satisfied": 1,
    "violated": 0,
    "error": -1,
    "timeout": -2,
    "invalid": -3,
    "nonsuccess": -4,
}

INT_TO_LTL_MC_STATUS = {v: k for k, v in LTL_MC_STATUS_TO_INT.items()}


class LTLMCStatus(Cat, CSV):
    def __init__(self, status: str) -> None:
        if status not in ["satisfied", "violated", "error", "timeout", "invalid", "nonsuccess"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def token(self, **kwargs) -> str:
        return self._status

    @classmethod
    def from_int(cls, i: int) -> "LTLMCStatus":
        return cls(status=INT_TO_LTL_MC_STATUS[i])

    def to_int(self) -> int:
        return LTL_MC_STATUS_TO_INT[self._status]

    @property
    def satisfied(self):
        return self._status == "satisfied"

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"satisfied": str(self.to_int())}

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["satisfied"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLMCStatus":
        # TODO change realizable to status in datasets
        return cls.from_int(int(fields["satisfied"]))

    @classmethod
    def from_token(cls, token: str, **kwargs) -> "LTLMCStatus":
        return cls(status=token)

    def to_pb2_LTLMCStatus(self, **kwargs):
        if self._status == "satisfied":
            return ltl_mc_pb2.LTLMCSTATUS_SATISFIED, self._status.upper()
        elif self._status == "violated":
            return ltl_mc_pb2.LTLMCSTATUS_VIOLATED, self._status.upper()
        elif self._status == "timeout":
            return ltl_mc_pb2.LTLMCSTATUS_TIMEOUT, self._status.upper()
        elif self._status == "error":
            return ltl_mc_pb2.LTLMCSTATUS_ERROR, self._status.upper()
        elif self._status == "invalid":
            return ltl_mc_pb2.LTLMCSTATUS_INVALID, self._status.upper()
        elif self._status == "nonsuccess":
            return ltl_mc_pb2.LTLMCSTATUS_NONSUCCESS, self._status.upper()
        else:
            return ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED, self._status.upper()

    @classmethod
    def from_pb2_LTLMCStatus(
        cls, pb2_LTLSpecification, detailed_status: str, **kwargs
    ) -> "LTLMCStatus":
        if pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_SATISFIED or (
            pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED
            and detailed_status.startswith("SATISFIED")
        ):
            return cls("satisfied")
        elif pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_VIOLATED or (
            pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED
            and detailed_status.startswith("VIOLATED")
        ):
            return cls("violated")
        elif pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_TIMEOUT or (
            pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED
            and detailed_status.startswith("TIMEOUT")
        ):
            return cls("timeout")
        elif pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_ERROR or (
            pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED
            and detailed_status.startswith("ERROR")
        ):
            return cls("error")
        elif pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_INVALID or (
            pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED
            and detailed_status.startswith("INVALID")
        ):
            return cls("invalid")
        elif pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_NONSUCCESS or (
            pb2_LTLSpecification == ltl_mc_pb2.LTLMCSTATUS_UNSPECIFIED
            and detailed_status.startswith("NONSUCCESS")
        ):
            return cls("nonsuccess")
        else:
            raise ValueError("Mapping LTLMCSTATUS not possible")
