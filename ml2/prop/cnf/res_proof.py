"""Resolution proof"""

import itertools
from typing import Dict, List, Tuple

from ...datasets.utils import from_csv_str, to_csv_str
from ...dtypes import CSV, Seq, String
from ...grpc.prop import prop_pb2
from ...registry import register_type
from .cnf_formula import Clause


class ResClause(Seq):
    def __init__(self, id: int) -> None:
        self.id = id
        self.clause = Clause()
        self.premises: List[int] = []

    def add_lit(self, lit: int, check_unique: bool = False) -> None:
        self.clause.add_lit(lit, check_unique)

    def add_premise(self, premise: int) -> None:
        self.premises.append(premise)

    @property
    def has_premises(self) -> bool:
        return len(self.premises) > 0

    def sort(self, lit_key=abs, premise_key=abs) -> None:
        if lit_key is not None:
            self.clause.sort(key=lit_key)
        if premise_key is not None:
            self.premises.sort(key=premise_key)

    def to_tokens(self, **kwargs) -> List[str]:
        return self.tracecheck_str.split()

    @property
    def tracecheck_str(self) -> str:
        return f"{self.id} {self.clause.to_dimacs_str()}{''.join([' ' + str(p) for p in self.premises])} 0"

    @classmethod
    def from_lists(cls, id: int, clause: List[int], premises: List[int] = None) -> "ResClause":
        rc = ResClause(id)
        for l in clause:
            rc.add_lit(l)
        if premises:
            for p in premises:
                rc.add_premise(p)
        return rc

    @classmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "ResClause":
        raise NotImplementedError()


class TraceCheckParseException(Exception):
    pass


@register_type
class ResProof(CSV, Seq, String):
    def __init__(self) -> None:
        self.res_clauses: List[ResClause] = []

    def add_res_clause(self, res_clause: ResClause) -> None:
        self.res_clauses.append(res_clause)

    def drop_core(self) -> None:
        self.res_clauses = [rc for rc in self.res_clauses if rc.has_premises]

    @property
    def pb(self) -> List[prop_pb2.ResClause]:
        return [
            prop_pb2.ResClause(id=rc.id, literals=rc.clause.lits, premises=rc.premises)
            for rc in self.res_clauses
        ]

    def sort(self, lit_key=abs, premise_key=abs, clause_key=None) -> None:
        [rc.sort(lit_key=lit_key, premise_key=premise_key) for rc in self.res_clauses]
        if clause_key is not None:
            self.res_clauses.sort(key=clause_key)

    def to_tokens(self, **kwargs) -> List[str]:
        return list(
            itertools.chain.from_iterable([rc.to_tokens(**kwargs) for rc in self.res_clauses])
        )

    def to_tokens_with_reward(self, **kwargs):
        tokens = []
        reward = []
        num_rcs = len(self.res_clauses)
        for i, rc in enumerate(self.res_clauses):
            rc_tokens = rc.to_tokens(**kwargs)
            tokens += rc_tokens
            reward += len(rc_tokens) * [str(num_rcs - i)]
        return tokens, reward

    def to_str(self, *, notation: str = "tracecheck") -> str:
        if notation == "tracecheck":
            return "".join([f"{rc.tracecheck_str}\n" for rc in self.res_clauses])
        elif notation == "tracecheck-sorted":
            return "".join(
                [f"{rc.tracecheck_str}\n" for rc in sorted(self.res_clauses, key=lambda c: c.id)]
            )
        else:
            raise ValueError(f"Invalid notation {notation}")

    def to_tracecheck_file(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            for rc in self.res_clauses:
                f.write(f"{rc.tracecheck_str}\n")

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"res_proof": to_csv_str(self.to_str())}

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "ResProof":
        return cls.from_tracecheck_str(tracecheck_str=from_csv_str(fields["res_proof"]))

    @classmethod
    def from_list(cls, proof: List[Tuple[int, List[int], List[int]]]) -> "ResProof":
        rp = cls()
        for id, c, p in proof:
            rp.add_res_clause(ResClause.from_lists(id, c, p))
        return rp

    @classmethod
    def from_pb(cls, pb: List[prop_pb2.ResClause]) -> "ResProof":
        return cls.from_list([(pb_rc.id, pb_rc.literals, pb_rc.premises) for pb_rc in pb])

    @classmethod
    def from_str(cls, s: str, notation: str = "tracecheck") -> "ResProof":
        if notation == "tracecheck":
            return cls.from_tracecheck_str(tracecheck_str=s)
        else:
            raise ValueError(f"Res proof notation {notation} not supported")

    @classmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "ResProof":
        return cls.from_str(" ".join(tokens))

    @classmethod
    def from_tracecheck_file(cls, filepath: str) -> "ResProof":
        with open(filepath, "r") as file:
            tracecheck_str = file.read()
        return cls.from_tracecheck_str(tracecheck_str=tracecheck_str)

    @classmethod
    def from_tracecheck_str(cls, tracecheck_str: str) -> "ResProof":
        rp = cls()
        next_token = "id"

        for l in tracecheck_str.rstrip().split("\n"):
            for next_str in l.split():
                if next_token == "id":
                    try:
                        id = int(next_str)
                    except ValueError:
                        raise TraceCheckParseException("Non-integer id")
                    next_res_clause = ResClause(id)
                    next_token = "literal"
                elif next_token == "literal":
                    try:
                        lit = int(next_str)
                    except ValueError:
                        raise TraceCheckParseException("Non-integer literal")
                    if lit == 0:
                        next_token = "premise"
                    else:
                        next_res_clause.add_lit(lit)
                elif next_token == "premise":
                    try:
                        premise = int(next_str)
                    except ValueError:
                        raise TraceCheckParseException("Non-integer permise")
                    if premise == 0:
                        rp.add_res_clause(next_res_clause)
                        next_token = "id"
                    else:
                        next_res_clause.add_premise(premise)
                else:
                    raise Exception("Invalid next token type")

        return rp
