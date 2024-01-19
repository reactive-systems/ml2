""""Assignment"""

import re
from typing import Dict, List, Tuple

from ..dtypes import CSV, Seq
from ..registry import register_type


@register_type
class Assignment(CSV, Seq):
    def __init__(self, assignment: Dict[str, bool] = None) -> None:
        self.assignment = assignment if assignment is not None else {}

    def to_str(
        self,
        assign_op: str = None,
        not_op: str = None,
        delimiter: str = ",",
        value_type: str = "num",
        sort_props: bool = False,
        **kwargs,
    ) -> str:
        return delimiter.join(
            self.assign_to_str(
                assign,
                assign_op=assign_op,
                not_op=not_op,
                value_type=value_type,
            )
            for assign in (
                sorted(self.assignment.items()) if sort_props else self.assignment.items()
            )
        )

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"assignment": self.to_str(**kwargs)}

    def to_tokens(
        self, assign_op: str = None, not_op: str = None, delimiter: str = None, **kwargs
    ) -> List[str]:
        if assign_op is None:
            if not_op is None:
                assign_op = " "
        else:
            assign_op = f" {assign_op} "

        if delimiter is None:
            delimiter = " "
        else:
            delimiter = f" {delimiter} "

        if not_op is not None:
            not_op = f" {not_op} "

        return self.to_str(
            assign_op=assign_op, not_op=not_op, delimiter=delimiter, **kwargs
        ).split()

    def filter_props(self, props: List[str]) -> None:
        self.assignment = {p: v for (p, v) in self.assignment.items() if p in props}

    def items(self):
        return self.assignment.items()

    def __contains__(self, key):
        return key in self.assignment

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.assignment == other.assignment
        return False

    def __getitem__(self, key):
        return self.assignment[key]

    def __setitem__(self, key, value):
        self.assignment[key] = value

    def __len__(self):
        return len(self.assignment)

    def __iter__(self):
        return iter(self.assignment)

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["assignment"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "Assignment":
        return cls.from_str(fields["assignment"], **kwargs)

    @classmethod
    def from_str(
        cls,
        s: str,
        assign_op: str = None,
        not_op: str = None,
        value_type: str = "num",
        delimiter: str = ",",
        **kwargs,
    ) -> "Assignment":
        assert not (delimiter is None or delimiter == "")

        if s.strip() == "":
            return cls(assignment={})

        split_assignment = s.split(delimiter)

        assignment = dict(
            cls.assign_from_str(assign, assign_op=assign_op, not_op=not_op, value_type=value_type)
            for assign in split_assignment
        )

        return cls(assignment=assignment)

    @classmethod
    def from_tokens(
        cls, tokens: List[str], not_op: str = None, delimiter: str = None, **kwargs
    ) -> "Assignment":
        if delimiter is None and not_op is None:
            delimiter = ","
            token_iter = iter(tokens)
            assign_str = ""
            for token in token_iter:
                assign_str = assign_str + delimiter if assign_str != "" else ""
                assign_str += token + next(token_iter)
            return cls.from_str(assign_str, not_op=not_op, delimiter=delimiter, **kwargs)
        else:
            return cls.from_str(" ".join(tokens), not_op=not_op, delimiter=delimiter, **kwargs)

    @staticmethod
    def assign_from_str(
        assign: str, assign_op: str = None, not_op: str = None, value_type: str = "num"
    ) -> Tuple[str, bool]:
        assert assign_op is None or not_op is None

        if not_op is not None:
            pat = "^\s*([0-9A-Za-z_]+)\s*$"
            not_pat = f"^\s*{not_op}\s*([0-9A-Za-z_]+)\s*$"
            if m := re.match(pat, assign):
                return (m.group(1), True)
            elif m := re.match(not_pat, assign):
                return (m.group(1), False)
            else:
                raise ValueError(f"Invalid assignment: {assign}")

        assign_op_pat = "" if assign_op is None else assign_op
        if value_type == "num":
            true_val_pat, false_val_pat = "1", "0"
        elif value_type == "bool":
            true_val_pat, false_val_pat = "(True|TRUE)", "(False|FALSE)"
        else:
            raise ValueError(f"Invalid value type: {value_type}")

        true_pat = f"^\s*([0-9A-Za-z_]+)\s*{assign_op_pat}\s*{true_val_pat}\s*$"
        false_pat = f"^\s*([0-9A-Za-z_]+)\s*{assign_op_pat}\s*{false_val_pat}\s*$"

        if m := re.match(true_pat, assign):
            return (m.group(1), True)
        elif m := re.match(false_pat, assign):
            return (m.group(1), False)
        else:
            raise ValueError(f"Invalid assignment: {assign}")

    @staticmethod
    def assign_to_str(
        assign: Tuple[str, bool],
        assign_op: str = None,
        not_op: str = None,
        value_type: str = "num",
    ) -> str:
        assert assign_op is None or not_op is None
        p, v = assign

        if not_op is not None:
            return p if v else f"{not_op} {p}"

        assign_op_str = "" if assign_op is None else assign_op
        if value_type == "num":
            true_str, false_str = "1", "0"
        elif value_type == "bool":
            true_str, false_str = "True", "False"
        else:
            raise ValueError(f"Invalid value type: {value_type}")

        return f"{p}{assign_op_str}{true_str if v else false_str}"
