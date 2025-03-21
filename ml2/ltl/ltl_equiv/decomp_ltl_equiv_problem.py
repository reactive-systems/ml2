"""Decomposed LTL equivalence problem"""

import hashlib
from typing import Dict, List, Optional, Tuple

from ...dtypes import CSV, Pair, Supervised
from ...registry import register_type
from ..ltl_spec import DecompLTLSpec
from .ltl_equiv_status import LTLEquivStatus


@register_type
class DecompLTLEquivProblem(CSV, Supervised[Pair[DecompLTLSpec, DecompLTLSpec], LTLEquivStatus]):

    def __init__(
        self,
        spec_fst: DecompLTLSpec,
        spec_snd: DecompLTLSpec,
        solution: LTLEquivStatus,
    ) -> None:
        self.spec_fst = spec_fst
        self.spec_snd = spec_snd
        self.solution = solution

    @property
    def input(self) -> Tuple[DecompLTLSpec, DecompLTLSpec]:
        return (self.spec_fst, self.spec_snd)

    @property
    def target(self) -> LTLEquivStatus:
        return self.solution

    def _to_csv_fields(
        self, notation: Optional[str] = None, suffix="", **kwargs
    ) -> Dict[str, str]:
        spec_1_fields = self.spec_fst.to_csv_fields(
            notation=notation, suffix=suffix + "_fst", **kwargs
        )
        spec_2_fields = self.spec_snd.to_csv_fields(
            notation=notation, suffix=suffix + "_snd", **kwargs
        )
        sol_fields = self.solution.to_csv_fields(suffix=suffix, **kwargs)
        return {**spec_1_fields, **spec_2_fields, **sol_fields}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.spec_fst)}\n{str(self.spec_snd)}\n{str(self.solution)}"

    @classmethod
    def _csv_field_header(cls, suffix="", **kwargs) -> List[str]:
        return list(
            set(
                DecompLTLSpec.csv_field_header(suffix=suffix + "_fst", **kwargs)
                + DecompLTLSpec.csv_field_header(suffix=suffix + "_snd", **kwargs)
                + LTLEquivStatus.csv_field_header(suffix=suffix, **kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(
        cls, fields: Dict[str, str], suffix="", **kwargs
    ) -> "DecompLTLEquivProblem":
        spec_fst = DecompLTLSpec.from_csv_fields(fields, suffix=suffix + "_fst", **kwargs)
        spec_snd = DecompLTLSpec.from_csv_fields(fields, suffix=suffix + "_snd", **kwargs)
        solution = LTLEquivStatus.from_csv_fields(fields, **kwargs)
        return cls(spec_fst, spec_snd, solution)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.spec_fst.cr_hash,
                        self.spec_snd.cr_hash,
                        self.solution._status,
                    )
                ).encode()
            ).hexdigest(),
            16,
        )
