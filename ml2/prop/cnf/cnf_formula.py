"""Propositional formula in conjunctive normal form"""

from typing import Dict, List

from ...datasets.utils import from_csv_str, to_csv_str
from ...dtypes import Seq
from ...grpc.prop import prop_pb2
from ...registry import register_type
from ..prop_formula import PropFormula


@register_type
class Clause(Seq):
    def __init__(self) -> None:
        self.lits: List[int] = []

    def add_lit(self, lit: int, check_unique: bool = False) -> None:
        if check_unique and (lit in self.lits or -lit in self.lits):
            raise ValueError("Clause already contains literal")
        self.lits.append(lit)

    @property
    def max_var(self) -> int:
        return abs(max(self.lits, key=abs))

    @property
    def num_lits(self) -> int:
        return len(self.lits)

    def sort(self, key=abs) -> None:
        self.lits.sort(key=key)

    def to_dimacs_str(self, zero_terminated: bool = True) -> str:
        dimacs_str = " ".join([str(l) for l in self.lits])
        if self.lits and zero_terminated:
            dimacs_str += " "
        if zero_terminated:
            dimacs_str += "0"
        return dimacs_str

    def to_str(self, **kwargs) -> str:
        return self.to_dimacs_str(**kwargs)

    def to_tokens(self, **kwargs) -> List[str]:
        return self.to_str(**kwargs).split()

    @classmethod
    def from_list(cls, clause: List[int], check_unique: bool = False) -> "Clause":
        c = cls()
        for l in clause:
            c.add_lit(l, check_unique)
        return c


class DIMACSParseException(Exception):
    pass


@register_type
class CNFFormula(PropFormula):
    def __init__(
        self, num_vars: int = None, symbol_table: List[str] = None, comments: List[str] = None
    ) -> None:
        self.clauses: List[Clause] = []
        self._num_vars = num_vars
        self.symbol_table = symbol_table
        self.comments = comments if comments else []

    def add_clause(self, clause: Clause) -> None:
        if self._num_vars and clause.max_var > self._num_vars:
            raise ValueError("Clause contains variable exceeding number of variables")
        self.clauses.append(clause)

    def __iter__(self):
        return iter(self.clauses)

    def __len__(self) -> int:
        return len(self.clauses)

    @property
    def num_clauses(self) -> int:
        return len(self.clauses)

    @property
    def num_vars(self) -> int:
        return self._num_vars if self._num_vars else max([c.max_var for c in self.clauses])

    @property
    def pb(self) -> prop_pb2.CNFFormula:
        pb_clauses = [prop_pb2.Clause(literals=c.lits) for c in self.clauses]
        return prop_pb2.CNFFormula(
            num_vars=self.num_vars,
            num_clauses=self.num_clauses,
            clauses=pb_clauses,
            comments=self.comments,
            symbol_table=self.symbol_table,
        )

    def sort(self, lit_key=abs, clause_key=None) -> None:
        if lit_key is not None:
            [c.sort(key=lit_key) for c in self.clauses]
        if clause_key is not None:
            self.clauses.sort(key=clause_key)

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"formula": to_csv_str(self.to_str(**kwargs))}

    def to_dimacs_file(self, filepath: str) -> None:
        with open(filepath, "w") as file:
            file.write(self.to_dimacs_str())

    def to_dimacs_str(
        self,
        numbered: bool = False,
        header: bool = True,
        comment: bool = True,
        sep: str = "\n",
        **kwargs,
    ) -> str:
        comment_str = "".join([f"c {co}\n" for co in self.comments])
        header_str = f"p cnf {self.num_vars} {self.num_clauses}\n"
        clause_str = "".join(
            [
                (f"{i + 1} " if numbered else "") + cl.to_dimacs_str(**kwargs) + "\n"
                for i, cl in enumerate(self.clauses)
            ]
        )
        result = (comment_str if comment else "") + (header_str if header else "") + clause_str
        if sep != "\n":
            return result.replace("\n", sep)
        return result

    def to_str(self, notation: str = "dimacs", **kwargs) -> str:
        if notation == "dimacs":
            return self.to_dimacs_str(**kwargs)
        else:
            raise ValueError(f"Invalid notation: {notation}")

    def to_tokens(self, **kwargs) -> List[str]:
        return self.to_str(**kwargs).split()

    @classmethod
    def from_clause_list(
        cls,
        clauses: List[Clause],
        num_vars: int = None,
        symbol_table: List[str] = None,
        comments: List[str] = None,
    ) -> "CNFFormula":
        p = cls(num_vars=num_vars, symbol_table=symbol_table, comments=comments)
        for c in clauses:
            p.add_clause(c)
        return p

    @classmethod
    def from_components(cls, components: List[Clause], **kwargs) -> "CNFFormula":
        return cls.from_clause_list(clauses=components, **kwargs)

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CNFFormula":
        return cls.from_dimacs_str(dimacs_str=from_csv_str(fields["formula"]))

    @classmethod
    def from_dimacs_file(cls, filepath: str) -> "CNFFormula":
        with open(filepath, "r") as file:
            dimacs_str = file.read()
        return cls.from_dimacs_str(dimacs_str=dimacs_str)

    @classmethod
    def from_dimacs_str(cls, dimacs_str: str) -> "CNFFormula":
        num_vars = -1
        num_clauses = -1
        clauses = []
        comments = []
        next_clause = Clause()
        for l in dimacs_str.rstrip().split("\n"):
            sl = l.split()
            # comment
            if sl[0] == "c":
                # comment
                comments.append(l[l.find("c") + 2 :])
            # header
            elif sl[0] == "p":
                if num_vars != -1:
                    raise DIMACSParseException("Second header found")
                if not (len(sl) == 4 and sl[1] == "cnf"):
                    raise DIMACSParseException("Invalid header")
                try:
                    num_vars = int(sl[2])
                except ValueError:
                    raise DIMACSParseException("Non-integer number of variables")
                try:
                    num_clauses = int(sl[3])
                except ValueError:
                    raise DIMACSParseException("Non-integer number of clauses")
            # clause
            else:
                for lit_str in sl:
                    try:
                        lit = int(lit_str)
                    except ValueError:
                        raise DIMACSParseException("Non-integer in clause")
                    if lit == 0:
                        clauses.append(next_clause)
                        next_clause = Clause()
                    else:
                        next_clause.add_lit(lit)

        if num_clauses == -1 or num_vars == -1:
            raise DIMACSParseException("No header found")

        if num_clauses != len(clauses):
            raise DIMACSParseException(
                "Specified number of clauses does not match parsed number of clauses"
            )

        return cls.from_clause_list(clauses=clauses, num_vars=num_vars, comments=comments)

    @classmethod
    def from_int_lists(
        cls,
        clauses: List[List[int]],
        num_vars: int = None,
        symbol_table: List[str] = None,
        comments: List[str] = None,
    ) -> "CNFFormula":
        return cls.from_clause_list(
            clauses=[Clause.from_list(lits) for lits in clauses],
            num_vars=num_vars,
            symbol_table=symbol_table,
            comments=comments,
        )

    @classmethod
    def from_pb(cls, pb: prop_pb2.CNFFormula) -> "CNFFormula":
        return cls.from_int_lists(
            clauses=[c.literals for c in pb.clauses],
            num_vars=pb.num_vars,
            symbol_table=pb.symbol_table,
            comments=pb.comments,
        )

    @classmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "CNFFormula":
        raise NotImplementedError()
