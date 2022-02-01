"""Status of an LTL synthesis problem"""

import enum


class LTLSynStatus(enum.Enum):
    REALIZABLE = "realizable"
    UNREALIZABLE = "unrealizable"
    TIMEOUT = "timeout"
    ERROR = "error"
