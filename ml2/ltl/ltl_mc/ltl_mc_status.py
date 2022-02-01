"""Status of an LTL model checking problem"""

import enum


class LTLMCStatus(enum.Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"
