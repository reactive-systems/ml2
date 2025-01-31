"""LTL equivalence problem"""

import hashlib
from typing import Dict, List, Optional, Tuple

from ...dtypes import CSVWithId, Pair, Supervised
from ..ltl_spec import LTLSpec
from .ltl_equiv_status import LTLEquivStatus


class LTLEquivProblem(
    CSVWithId,
    Supervised[Pair[LTLSpec, LTLSpec], LTLEquivStatus],
):

    def __init__(
        self,
        spec_fst: LTLSpec,
        spec_snd: LTLSpec,
        solution: LTLEquivStatus,
    ) -> None:
        self.spec_fst = spec_fst
        self.spec_snd = spec_snd
        self.solution = solution

    @property
    def input(self) -> Tuple[LTLSpec, LTLSpec]:
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
                LTLSpec.csv_field_header(suffix=suffix + "_fst", **kwargs)
                + LTLSpec.csv_field_header(suffix=suffix + "_snd", **kwargs)
                + LTLEquivStatus.csv_field_header(suffix=suffix, **kwargs)
            )
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], suffix="", **kwargs) -> "LTLEquivProblem":
        spec_fst = LTLSpec.from_csv_fields(fields, suffix=suffix + "_fst", **kwargs)
        spec_snd = LTLSpec.from_csv_fields(fields, suffix=suffix + "_snd", **kwargs)
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
