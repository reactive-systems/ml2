"""Status of an LTL satisfiability problem"""

import enum


class LTLSatStatus(enum.Enum):
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"
