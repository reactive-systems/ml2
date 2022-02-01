"""Trace"""

import re


class Trace(object):
    def __init__(self, cycle: list, prefix: list = None):
        self.prefix = [[ap.strip() for ap in p] for p in prefix] if prefix else []
        self.cycle = [[ap.strip() for ap in p] for p in cycle]

    def to_str(self, spot: bool = False) -> str:
        prefix_str = " ; ".join([" , ".join(p) for p in self.prefix])
        cycle_str = " ; ".join([" , ".join(p) for p in self.cycle])
        return (
            f'{prefix_str}{" ; " if self.prefix else ""}{"cycle" if spot else ""}{{ {cycle_str} }}'
        )

    @classmethod
    def from_str(cls, trace: str, spot: bool = False):
        """
        Constructs Trace objects from strings where positions are seperated by semicolons, propositions by commas,
        and cycles by curly brackets, e.g. a , b ; a , ! b ; { c , b ; a }
        """
        if spot:
            p = re.compile(r"^([A-Za-z0-9!,;\s]*)cycle\{([A-Za-z0-9!,;\s]*)\}$")
        else:
            p = re.compile(r"^([A-Za-z0-9!,;\s]*)\{([A-Za-z0-9!,;\s]*)\}$")

        m = p.match(trace)
        if m:
            prefix = [p.split(",") for p in m.group(1).split(";")[:-1]] if m.group(1) else None
            cycle = [p.split(",") for p in m.group(2).split(";")]
            return cls(cycle=cycle, prefix=prefix)
        else:
            return None

    @classmethod
    def from_aalta_str(cls, trace: str):
        p = re.compile(r"^(.*)\(\n(.*)\n\)\^w\n$", re.MULTILINE | re.DOTALL)
        m = p.match(trace)
        if m:
            prefix_str = m.group(1) if m.group(1) else ""
            cycle_str = m.group(2)
            for c in ["(", ")", "{", ",}", "true"]:
                prefix_str = prefix_str.replace(c, "")
                cycle_str = cycle_str.replace(c, "")
            prefix = (
                [p.split(",") if p else [] for p in prefix_str.split("\n")[:-1]]
                if prefix_str
                else []
            )
            cycle = [p.split(",") if p else [] for p in cycle_str.split("\n")]
            return cls(cycle=cycle, prefix=prefix)
        else:
            return None
