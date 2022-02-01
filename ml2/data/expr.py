"""Expression classes"""

from enum import Enum


class ExprNotation(Enum):
    PREFIX = "prefix"
    INFIX = "infix"
    INFIXNOPARS = "infix-no-pars"
