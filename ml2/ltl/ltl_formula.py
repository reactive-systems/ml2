"""LTL formula"""

import hashlib
from typing import Any, Dict, List, Optional

from ..dtypes.binary_ast import BinaryAST
from ..dtypes.binary_expr import BinaryExpr
from ..dtypes.csv_dtype_with_id import CSVWithId
from ..dtypes.decomp_binary_expr import DecompBinaryExpr
from ..grpc.ltl import ltl_pb2
from ..registry import register_type
from ..utils.list_utils import join_lists
from .ltl_lexer import lex_ltl
from .ltl_parser import parse_infix_ltl, parse_prefix_ltl


@register_type
class LTLFormula(BinaryExpr, CSVWithId):
    def __init__(
        self,
        ast: BinaryAST = None,
        formula: str = None,
        notation: str = None,
        tokens: List[str] = None,
    ) -> None:
        super().__init__(ast=ast, formula=formula, notation=notation, tokens=tokens)

    @property
    def cr_hash(self) -> int:
        return int(hashlib.sha3_224(self.to_str("prefix").encode()).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.ast == other.ast
        return False

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {"formula": self.to_str(notation=notation)}
        return fields

    def to_pb2_LTLFormula(self, notation: Optional[str] = None, **kwargs):
        if notation is None:
            if self._notation is None:
                notation = "infix"
            else:
                notation = self._notation

        return ltl_pb2.LTLFormula(
            formula=self.to_str(notation=notation, **kwargs), notation=notation
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["formula"]

    @classmethod
    def _from_csv_fields(
        cls, fields: Dict[str, str], notation: str = None, **kwargs
    ) -> "LTLFormula":
        return cls(formula=fields["formula"], notation=notation)

    @classmethod
    def from_pb2_LTLFormula(cls, pb2_LTLFormula, **kwargs) -> "LTLFormula":
        return cls.from_str(
            formula=pb2_LTLFormula.formula, notation=pb2_LTLFormula.notation, **kwargs
        )

    @staticmethod
    def lex(expr: str) -> List[str]:
        return lex_ltl(expr)

    @staticmethod
    def parse(expr: str, notation: str = "infix") -> BinaryAST:
        if notation == "infix":
            return parse_infix_ltl(expr)
        elif notation == "prefix":
            return parse_prefix_ltl(expr)
        else:
            raise ValueError(f"Invalid notation {notation}")


@register_type
class DecompLTLFormula(DecompBinaryExpr, LTLFormula):
    BINARY_EXPR_TYPE = LTLFormula

    def __init__(
        self,
        sub_exprs: Optional[List[LTLFormula]] = None,
    ):
        DecompBinaryExpr.__init__(self, sub_exprs=sub_exprs)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str((sorted([sub.cr_hash for sub in self.sub_exprs]))).encode()
            ).hexdigest(),
            16,
        )

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        return {
            "formulas": ",".join([f.to_str(notation=notation) for f in self.sub_exprs]),
        }

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["formulas"]

    @classmethod
    def _from_csv_fields(
        cls, fields: Dict[str, str], notation: str = None, **kwargs
    ) -> "DecompLTLFormula":
        d = fields.copy()
        if "formulas" in d and d["formulas"] != "":
            d["formulas"] = d["formulas"].split(",")
        else:
            d["formulas"] = None
        return cls.from_dict(d=d, notation=notation, **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], notation: str = None, **kwargs) -> "DecompLTLFormula":
        return cls(
            sub_exprs=[LTLFormula(formula=f, notation=notation) for f in d["formulas"]]
            if "formulas" in d and d["formulas"] is not None
            else None
        )

    @classmethod
    def decompose_formula(cls, formula: LTLFormula) -> "DecompLTLFormula":
        def collect_decomposed(ast: BinaryAST) -> List[BinaryAST]:
            if ast.label != "&" and ast.label != "&&":
                return [ast]
            else:
                assert len(ast) == 2
                return collect_decomposed(ast.lhs) + collect_decomposed(ast.rhs)

        return cls(sub_exprs=[LTLFormula.from_ast(a) for a in collect_decomposed(formula.ast)])

    @staticmethod
    def comp_asts(asts: List[BinaryAST]) -> BinaryAST:
        if len(asts) == 0:
            return None
        comp_ast = asts[0]
        for ast in asts[1:]:
            comp_ast = BinaryAST("&", comp_ast, ast)
        return comp_ast

    @staticmethod
    def comp_strs(strs: List[str], notation: str = None) -> str:
        if notation == "infix":
            return " & ".join([f"( {a} )" for a in strs])
        elif notation == "infix-no-pars":
            return " & ".join(strs)
        elif notation == "prefix":
            return "& " * max(len(strs) - 1, 0) + " ".join(strs)
        else:
            raise ValueError(f"Unknown notation {notation}")

    @staticmethod
    def comp_token_lists(token_lists: List[List[str]], notation: str = None) -> List[str]:
        if notation == "infix":
            return join_lists("&", [["("] + l + [")"] for l in token_lists])
        elif notation == "infix-no-pars":
            return join_lists("&", token_lists)
        elif notation == "prefix":
            return ["&"] * max(len(token_lists) - 1, 0) + [t for l in token_lists for t in l]
        else:
            raise ValueError(f"Unknown notation {notation}")
