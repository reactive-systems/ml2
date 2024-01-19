"""Status of an LTL synthesis problem"""

from ...grpc.ltl import ltl_syn_pb2
from ...registry import register_type
from .ltl_real_status import LTLRealStatus

STATUS_TO_INT = {
    "realizable": 1,
    "unrealizable": 0,
    "error": -1,
    "timeout": -2,
    "nonsuccess": -3,
}

INT_TO_STATUS = {v: k for k, v in STATUS_TO_INT.items()}


@register_type
class LTLSynStatus(LTLRealStatus):
    int_to_status = INT_TO_STATUS
    status_to_int = STATUS_TO_INT

    def __init__(self, status: str) -> None:
        if status not in ["realizable", "unrealizable", "timeout", "error", "nonsuccess"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    def to_pb2_LTLSynStatus(self, **kwargs):
        if self._status == "realizable":
            return ltl_syn_pb2.LTLSYNSTATUS_REALIZABLE, self._status.upper()
        elif self._status == "unrealizable":
            return ltl_syn_pb2.LTLSYNSTATUS_UNREALIZABLE, self._status.upper()
        elif self._status == "timeout":
            return ltl_syn_pb2.LTLSYNSTATUS_TIMEOUT, self._status.upper()
        elif self._status == "error":
            return ltl_syn_pb2.LTLSYNSTATUS_ERROR, self._status.upper()
        elif self._status == "nonsuccess":
            return ltl_syn_pb2.LTLSYNSTATUS_NONSUCCESS, self._status.upper()
        else:
            return ltl_syn_pb2.LTLSYNSTATUS_UNSPECIFIED, self._status.upper()

    @classmethod
    def from_pb2_LTLSynStatus(
        cls, pb2_LTLSynStatus, detailed_status: str, **kwargs
    ) -> "LTLSynStatus":
        if pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_REALIZABLE or (
            pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_UNSPECIFIED
            and detailed_status.startswith("REALIZABLE")
        ):
            return cls("realizable")
        elif pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_UNREALIZABLE or (
            pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_UNSPECIFIED
            and detailed_status.startswith("UNREALIZABLE")
        ):
            return cls("unrealizable")
        elif pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_TIMEOUT or (
            pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_UNSPECIFIED
            and detailed_status.startswith("TIMEOUT")
        ):
            return cls("timeout")
        elif pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_ERROR or (
            pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_UNSPECIFIED
            and detailed_status.startswith("ERROR")
        ):
            return cls("error")
        elif pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_NONSUCCESS or (
            pb2_LTLSynStatus == ltl_syn_pb2.LTLSYNSTATUS_UNSPECIFIED
            and detailed_status.startswith("NONSUCCESS")
        ):
            return cls("nonsuccess")
        else:
            raise ValueError("Mapping LTLSYNSTATUS not possible")
