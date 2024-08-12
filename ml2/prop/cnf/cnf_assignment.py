"""CNF assignment"""

from typing import Dict, List

from ...dtypes import CSV, Seq
from ...registry import register_type


@register_type
class CNFAssignment(CSV, Seq):
    def __init__(self, assignment: List[int]) -> None:
        self.assignment = assignment

    def to_str(self, **kwargs) -> str:
        return "v " + " ".join([str(v) for v in self.assignment]) + " 0"

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"assignment": self.to_str(**kwargs)}

    def to_tokens(self, **kwargs) -> List[str]:
        return [str(v) for v in self.assignment]

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["assignment"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFAssignment":
        return cls.from_str(fields["assignment"], **kwargs)

    @classmethod
    def from_str(cls, s: str, **kwargs) -> "CNFAssignment":
        assignment = []
        read_zero = False
        for l in s.split("\n"):
            if not l.startswith("v"):
                raise ValueError("New line does not start with v")
            for v in l.split(" ")[1:]:
                if v == "0":
                    read_zero = True
                    break
                assignment.append(int(v))
            if read_zero:
                break
        if read_zero:
            return cls(assignment=assignment)
        else:
            raise ValueError("Assignment string does not end with zero")

    @classmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "CNFAssignment":
        return cls(assignment=[int(v) for v in tokens])
