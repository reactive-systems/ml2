"""Tokenizer that encodes a decomposed expression into a decomposed sequence encoding with positional encoding"""


from typing import Any, Dict, Generic, TypeVar

from ...dtypes import BinaryExpr, DecompBinaryExpr
from ...registry import register_type
from ..decomp_dtype_tokenizers import DecompDTypeToDecompSeqPosTokenizer
from ..expr_tokenizers import ExprToSeqTPETokenizer

T = TypeVar("T", bound=DecompBinaryExpr)


@register_type
class DecompExprToDecompSeqTPETokenizer(
    DecompDTypeToDecompSeqPosTokenizer[T, BinaryExpr], Generic[T]
):
    def __init__(self, sub_tokenizer: ExprToSeqTPETokenizer, **kwargs):
        super().__init__(sub_tokenizer=sub_tokenizer, **kwargs)

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_num_sub_exprs(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            if "num_sub_exprs" not in config:
                # TODO check if key should be left in config, would allow to construct superclass of config
                config["num_sub_exprs"] = config.pop("num_sub_seqs")

        return [postprocess_num_sub_exprs] + super().config_postprocessors()

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_num_sub_exprs(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "num_sub_exprs" in config:
                config["num_sub_seqs"] = config.pop("num_sub_exprs")

        return [preprocess_num_sub_exprs] + super().config_preprocessors()
