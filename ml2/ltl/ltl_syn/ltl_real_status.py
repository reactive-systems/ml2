"""Status of an LTL realizability problem"""

from typing import Dict, List, Type, TypeVar

from ...dtypes import CSV, Cat
from ...grpc.ltl import ltl_syn_pb2

T = TypeVar("T", bound="LTLRealStatus")


STATUS_TO_INT = {
    "realizable": 1,
    "unrealizable": 0,
}

INT_TO_STATUS = {v: k for k, v in STATUS_TO_INT.items()}


class LTLRealStatus(Cat, CSV):
    int_to_status = INT_TO_STATUS
    status_to_int = STATUS_TO_INT

    def __init__(self, status: str) -> None:
        if status not in ["realizable", "unrealizable"]:
            raise ValueError(f"Invalid status {status}")
        self._status = status

    @property
    def realizable(self):
        if self._status not in ["realizable", "unrealizable"]:
            raise ValueError(f"Cannot determine realizability of status {self._status}")
        return self._status == "realizable"

    def token(self, **kwargs) -> str:
        return self._status

    def to_int(self) -> int:
        return self.status_to_int[self._status]

    def to_pb2_LTLRealStatus(self, **kwargs):
        if self._status == "realizable":
            return ltl_syn_pb2.LTLREALSTATUS_REALIZABLE, self._status.upper()
        elif self._status == "unrealizable":
            return ltl_syn_pb2.LTLREALSTATUS_UNREALIZABLE, self._status.upper()
        else:
            return ltl_syn_pb2.LTLREALSTATUS_UNSPECIFIED, self._status.upper()

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"realizable": str(self.to_int())}

    @classmethod
    def from_int(cls: Type[T], i: int) -> T:
        return cls(status=cls.int_to_status[i])

    @classmethod
    def from_realizable(cls: Type[T], b: bool) -> T:
        return cls.from_int(int(b))

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["realizable"]

    @classmethod
    def _from_csv_fields(cls: Type[T], fields: Dict[str, str], **kwargs) -> T:
        # TODO change realizable to status in datasets
        return cls.from_int(int(fields["realizable"]))

    @classmethod
    def from_token(cls: Type[T], token: str, **kwargs) -> T:
        return cls(status=token)

    @classmethod
    def from_pb2_LTLRealStatus(
        cls, pb2_LTLRealStatus, detailed_status: str, **kwargs
    ) -> "LTLRealStatus":
        if pb2_LTLRealStatus == ltl_syn_pb2.LTLREALSTATUS_REALIZABLE or (
            pb2_LTLRealStatus == ltl_syn_pb2.LTLREALSTATUS_UNSPECIFIED
            and detailed_status.startswith("REALIZABLE")
        ):
            return cls("realizable")
        elif pb2_LTLRealStatus == ltl_syn_pb2.LTLREALSTATUS_UNREALIZABLE or (
            pb2_LTLRealStatus == ltl_syn_pb2.LTLREALSTATUS_UNSPECIFIED
            and detailed_status.startswith("UNREALIZABLE")
        ):
            return cls("unrealizable")
        else:
            raise ValueError("Mapping LTLSYNSTATUS not possible")
