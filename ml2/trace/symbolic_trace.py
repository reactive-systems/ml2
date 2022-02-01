"""Symbolic Trace"""

import re

from typing import List

from ..prop import PropFormula


class SymbolicTrace(object):
    def __init__(self, cycle: list, prefix: list = None, notation: str = None):
        self.prefix = [p.strip() for p in prefix] if prefix else []
        self.cycle = [p.strip() for p in cycle]
        self._notation = notation

    def to_str(self, notation: str = None, spot: bool = False) -> str:
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

    def tokens(self, notation: str = None) -> List[str]:
        prefix_tokens = [
            t
            for p in self.prefix
            for t in PropFormula.from_str(p, self._notation).tokens(notation) + [";"]
        ]
        cycle_tokens = [
            t
            for p in self.cycle
            for t in PropFormula.from_str(p, self._notation).tokens(notation) + [";"]
        ][
            :-1
        ]  # remove last semicolon
        return prefix_tokens + ["{"] + cycle_tokens + ["}"]

    @classmethod
    def from_str(cls, trace: str, notation: str = "infix", spot: bool = False):
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
            return None

    @classmethod
    def from_tokens(cls, tokens: List[str], notation: str = "infix", spot: bool = False):
        return cls.from_str(" ".join(tokens), notation=notation, spot=spot)
