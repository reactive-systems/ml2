"""Propositional formula"""

from typing import Dict, List

from ..dtypes import CSV, BinaryAST, BinaryExpr
from ..registry import register_type
from .prop_lexer import lex_prop
from .prop_parser import parse_infix_prop, parse_prefix_prop


@register_type
class PropFormula(BinaryExpr, CSV):
    def __init__(
        self,
        ast: BinaryAST = None,
        formula: str = None,
        notation: str = None,
        tokens: List[str] = None,
    ) -> None:
        super().__init__(ast=ast, formula=formula, notation=notation, tokens=tokens)

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {"formula": self.to_str(notation=notation)}
        if notation is not None:
            fields["notation"] = notation.value
        return fields

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["formula", "notation"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "PropFormula":
        return cls(formula=fields["formula"], notation=fields.get("notation", "infix"))

    @staticmethod
    def lex(expr: str) -> List[str]:
        return lex_prop(expr)

    @staticmethod
    def parse(expr: str, notation: str = "infix") -> BinaryAST:
        if notation == "infix":
            return parse_infix_prop(expr)
        elif notation == "prefix":
            return parse_prefix_prop(expr)
        else:
            raise ValueError(f"Invalid notation {notation}")
