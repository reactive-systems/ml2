"""Symbolic Trace"""

import re
from typing import Dict, List

from ..prop.prop_formula import PropFormula
from ..registry import register_type
from .trace import Trace


@register_type
class SymbolicTrace(Trace):
    def __init__(self, cycle: List[str], prefix: List[str] = None, notation: str = None) -> None:
        self.prefix = [p.strip() for p in prefix] if prefix else []
        self.cycle = [p.strip() for p in cycle]
        self._notation = notation

    def to_str(self, notation: str = None, spot: bool = False, **kwargs) -> str:
        if not notation or notation == self._notation:
            prefix_str = " ; ".join(self.prefix)
            cycle_str = " ; ".join(self.cycle)
        else:
            prefix_str = " ; ".join(
                [PropFormula.from_str(p, self._notation).to_str(notation) for p in self.prefix]
            )
            cycle_str = " ; ".join(
                [PropFormula.from_str(c, self._notation).to_str(notation) for c in self.cycle]
            )

        return (
            f'{prefix_str}{" ; " if self.prefix else ""}{"cycle" if spot else ""}{{ {cycle_str} }}'
        )

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {"trace": self.to_str(notation)}
        return fields

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["trace"]

    def to_tokens(self, notation: str = None, **kwargs) -> List[str]:
        prefix_tokens = [
            t
            for p in self.prefix
            for t in PropFormula.from_str(p, self._notation).to_tokens(notation) + [";"]
        ]
        cycle_tokens = [
            t
            for p in self.cycle
            for t in PropFormula.from_str(p, self._notation).to_tokens(notation) + [";"]
        ][
            :-1
        ]  # remove last semicolon
        return prefix_tokens + ["{"] + cycle_tokens + ["}"]

    @classmethod
    def _from_csv_fields(
        cls, fields: Dict[str, str], notation: str = None, **kwargs
    ) -> "SymbolicTrace":
        return cls.from_str(fields["trace"], notation=notation)

    @classmethod
    def from_str(
        cls, trace: str, notation: str = "infix", spot: bool = False, **kwargs
    ) -> "SymbolicTrace":
        if spot:
            p = re.compile(r"^([A-Za-z0-9&|!;()\s]*)cycle\{([A-Za-z0-9&|!;()\s]*)\}$")
        else:
            p = re.compile(r"^([A-Za-z0-9&|!;()\s]*)\{([A-Za-z0-9&|!;()\s]*)\}$")

        m = p.match(trace)
        if m:
            prefix = m.group(1).split(";")[:-1] if m.group(1) else None
            cycle = m.group(2).split(";")
            return cls(cycle=cycle, prefix=prefix, notation=notation)
        else:
            raise Exception(f"Syntax error for trace string {trace}")

    @classmethod
    def from_tokens(
        cls,
        tokens: List[str],
        notation: str = "infix",
        spot: bool = False,
        **kwargs,
    ) -> "SymbolicTrace":
        return cls.from_str(" ".join(tokens), notation=notation, spot=spot, **kwargs)
