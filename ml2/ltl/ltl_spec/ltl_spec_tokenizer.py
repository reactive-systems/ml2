"""LTL specification tokenizer"""


from copy import deepcopy
from typing import Type

from ...registry import register_type
from ...tokenizers import ExprToSeqTokenizer, ExprToSeqTPETokenizer
from .ltl_spec import LTLSpec


class LTLSpecToSeqTokenizer(ExprToSeqTokenizer[LTLSpec]):
    def __init__(self, rename_aps: bool = False, dtype: Type[LTLSpec] = LTLSpec, **kwargs):
        self.rename_aps = rename_aps
        super().__init__(dtype=dtype, **kwargs)

    def preprocess_sample(self, x: LTLSpec) -> LTLSpec:
        if self.rename_aps:
            x = deepcopy(x)
            x.rename_aps(random=False)
        return x


@register_type
class LTLSpecToSeqTPETokenizer(ExprToSeqTPETokenizer[LTLSpec]):
    def __init__(self, rename_aps: bool = False, dtype: Type[LTLSpec] = LTLSpec, **kwargs):
        self.rename_aps = rename_aps
        super().__init__(dtype=dtype, **kwargs)

    def preprocess_sample(self, x: LTLSpec) -> LTLSpec:
        if self.rename_aps:
            x = deepcopy(x)
            x.rename_aps(random=False)
        return x
