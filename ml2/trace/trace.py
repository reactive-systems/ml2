"""Trace"""

import hashlib
import re
from itertools import chain
from typing import Dict, List

from ..dtypes import CSVWithId, Seq
from ..prop import Assignment
from ..registry import register_type


@register_type
class Trace(CSVWithId, Seq):
    def __init__(self, cycle: List[Assignment], prefix: List[Assignment] = None) -> None:
        self.prefix = prefix if prefix else []
        self.cycle = cycle

    def to_str(self, notation: str = "standard", **kwargs) -> str:
        prefix_str = " ; ".join(
            [p.to_str(not_op="!", delimiter=" , ", **kwargs) for p in self.prefix]
        )
        cycle_str = " ; ".join(
            [p.to_str(not_op="!", delimiter=" , ", **kwargs) for p in self.cycle]
        )
        if notation == "standard":
            return f'{prefix_str}{" ; " if self.prefix else ""}{{ {cycle_str} }}'
        elif notation == "spot":
            return f'{prefix_str}{" ; " if self.prefix else ""}cycle{{ {cycle_str} }}'
        else:
            raise ValueError(f"Unsupported notation: {notation}")

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        fields = {"trace": self.to_str(**kwargs)}
        return fields

    def to_tokens(self, **kwargs) -> List[str]:
        return self.to_str(**kwargs).split(" ")

    def complete_by_predecessor(self) -> None:
        for i in range(1, len(self)):
            for p, v in self[i - 1].items():
                if p not in self[i]:
                    self[i][p] = v

    def filter_props(self, props: List[str]) -> None:
        for assign in self:
            assign.filter_props(props)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.cycle == other.cycle and self.prefix == other.prefix
        return False

    def __getitem__(self, key):
        if key < len(self.prefix):
            return self.prefix[key]
        return self.cycle[key - len(self.prefix)]

    def __setitem__(self, key, value):
        if key < len(self.prefix):
            self.prefix[key] = value
        self.cycle[key - len(self.prefix)] = value

    def __len__(self):
        return len(self.prefix) + len(self.cycle)

    def __iter__(self):
        return chain(iter(self.prefix), iter(self.cycle))

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["trace"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "Trace":
        return cls.from_str(fields["trace"], **kwargs)

    @classmethod
    def from_aalta_str(cls, trace: str) -> "Trace":
        p = re.compile(r"^(.*)\(\n(.*)\n\)\^w\n$", re.MULTILINE | re.DOTALL)
        m = p.match(trace)
        if m:
            prefix_str = m.group(1) if m.group(1) else ""
            cycle_str = m.group(2)
            for c in ["(", ")", "{", ",}", "true"]:
                prefix_str = prefix_str.replace(c, "")
                cycle_str = cycle_str.replace(c, "")
            prefix = (
                [
                    Assignment.from_str(p, not_op="!", delimiter=",")
                    for p in prefix_str.split("\n")[:-1]
                ]
                if prefix_str
                else []
            )
            cycle = [
                Assignment.from_str(p, not_op="!", delimiter=",") for p in cycle_str.split("\n")
            ]
            return cls(cycle=cycle, prefix=prefix)
        raise Exception(f"Syntax error for aalta string: {trace}")

    @classmethod
    def from_nusmv_str(cls, trace: str, complete: bool = True) -> "Trace":
        cycle_split = trace.split("-- Loop starts here\n", 1)
        if len(cycle_split) == 2:
            prefix_str = cycle_split[0]
            cycle_str = cycle_split[1]
            # Due to NuSMV bug
            cycle_str = cycle_str.replace("-- Loop starts here\n", "")
        elif len(cycle_split) == 1:
            prefix_str = ""
            cycle_str = cycle_split[0]
        else:
            raise Exception(f"Syntax error for nusmv string: {trace}")

        prefix = (
            [
                Assignment.from_str(p.strip(), assign_op="=", delimiter="\n", value_type="bool")
                for p in re.split(r"\s*-> State: [0-9.]+ <-", prefix_str)[1:]
            ]
            if prefix_str
            else []
        )

        cycle = [
            Assignment.from_str(p.strip(), assign_op="=", delimiter="\n", value_type="bool")
            for p in re.split(r"\s*-> State: [0-9.]+ <-", cycle_str)[1:]
        ]

        return cls(cycle=cycle, prefix=prefix)

    @classmethod
    def from_generic_str(
        cls,
        trace: str,
        prop_delimiter: str = ",",
        pos_delimiter: str = ";",
        cycle_delimiter: str = "",
        **kwargs,
    ) -> "Trace":
        if not (m := re.match("^(.*)" + cycle_delimiter + "\{(.*)\}$", trace)):
            raise Exception(f"Syntax error for spot string: {trace}")

        prefix = (
            [
                Assignment.from_str(a, not_op="!", delimiter=prop_delimiter)
                for a in m.group(1).split(pos_delimiter)[:-1]
            ]
            if m.group(1)
            else None
        )

        cycle = [
            Assignment.from_str(a, not_op="!", delimiter=prop_delimiter)
            for a in m.group(2).split(pos_delimiter)
        ]

        return cls(cycle=cycle, prefix=prefix)

    @classmethod
    def from_str(cls, trace: str, notation: str = "standard", **kwargs) -> "Trace":
        """Constructs Trace object from strings in various notations.

        The following notations are supported:
        aalta: {a,(! b),}\n(\n{b,c,}\n{a,}\n)^w\n
        nusmv: -> State: 1.1 <-\n    a = TRUE\n    b = FALSE\n  -- Loop starts here\n  -> State: 1.2 <-\n    b = TRUE\n    c=TRUE\n  -> State: 1.3 <-\n    a = TRUE\n  -> State: 1.4 <-\n
        spot: a & ! b ; cycle{ b & c ; a }
        standard: a , ! b ; { b , c ; a }
        """

        if notation == "aalta":
            return cls.from_aalta_str(trace, **kwargs)
        elif notation == "nusmv":
            return cls.from_nusmv_str(trace, **kwargs)
        elif notation == "spot":
            return cls.from_generic_str(
                trace, prop_delimiter="&", cycle_delimiter="cycle", **kwargs
            )
        elif notation == "standard":
            return cls.from_generic_str(trace, **kwargs)
        else:
            raise ValueError(f"Invalid notation: {notation}")

    @classmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "Trace":
        return cls.from_str(" ".join(tokens), **kwargs)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(self.to_str().encode()).hexdigest(),
            16,
        )
