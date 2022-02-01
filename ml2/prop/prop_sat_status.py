"""Status of an propositional satisfiability problem"""

import enum


class PropSatStatus(enum.Enum):
    SAT = "sat"
    UNSAT = "unsat"
    TIMEOUT = "timeout"
    ERROR = "error"


class PropValidStatus(enum.Enum):
    VALID = "valid"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"
