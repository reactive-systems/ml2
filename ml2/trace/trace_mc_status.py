"""Status of an trace model checking problem"""

import enum


class TraceMCStatus(enum.Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"
