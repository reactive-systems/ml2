"""Tokenizer that encodes a decomposed expression pair into a decomposed sequence encoding with tree positional encoding"""

from typing import Any, Dict, Generator, Generic, List, Type, TypeVar

from ...dtypes import BinaryExpr, DecompBinaryExprPair
from ...registry import register_type
from ..to_decomp_seq_pos_tokenizer import DecompSeqPosEncoding, ToDecompSeqPosTokenizer
from ..to_seq_pos_tokenizer import SeqPosEncoding, ToSeqPosTokenizer
from ..tokenizer import PAD_TOKEN, TokenizationException

S = TypeVar("S", bound=BinaryExpr)
DS = DecompBinaryExprPair[S]
T = TypeVar("T", bound=DS)


@register_type
class DecompExprPairToDecompSeqTPETokenizer(ToDecompSeqPosTokenizer[S, T], Generic[S, T]):
    def __init__(
        self,
        dtype: Type[T],
        sub_tokenizer: ToSeqPosTokenizer[S],
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            sub_tokenizer=sub_tokenizer,
            **kwargs,
        )

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_num_sub_exprs(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            if "num_sub_exprs" not in config:
                config["num_sub_exprs"] = config.pop("num_sub_seqs")

        return [postprocess_num_sub_exprs] + super().config_postprocessors()

    def encode(self, data: T, **kwargs) -> DecompSeqPosEncoding:
        data = self.preprocess_sample(data)

        if len(data[0]) + len(data[1]) > self.num_sub_seqs:
            raise TokenizationException("Too many sub-expressions")
        encs = []
        # TODO fix hack to differentiate between first and second component
        self.sub_tokenizer.start = True
        for sub_expr in data[0].sub_exprs:
            encs.append(self.sub_tokenizer.encode(sub_expr, **kwargs))
        self.sub_tokenizer.start = False
        for sub_expr in data[1].sub_exprs:
            encs.append(self.sub_tokenizer.encode(sub_expr, **kwargs))
        for _ in range(len(data[0]) + len(data[1]), self.num_sub_seqs):
            encs.append(
                SeqPosEncoding(
                    tokens=[],
                    pad_tokens=[PAD_TOKEN] * self.sub_tokenizer.pad,
                    ids=self.sub_tokenizer.vocabulary.tokens_to_ids(
                        [PAD_TOKEN] * self.sub_tokenizer.pad
                    ),
                    pos_enc=[[] * self.sub_tokenizer.pad],
                    pad_pos_enc=[[0] * self.sub_tokenizer.pos_pad] * self.sub_tokenizer.pad,
                )
            )
        return DecompSeqPosEncoding(seq_pos_encs=encs)

    def decode(self, decomp_ids: List[List[int]], **kwargs) -> T:
        raise NotImplementedError()

    def preprocess_sample(self, x: T) -> T:
        return x

    def build_vocabulary(
        self,
        generator: Generator[T, None, None],
        add_start: bool = True,
        add_eos: bool = False,
        add_pad: bool = False,
        **kwargs,
    ) -> None:
        def sub_comp_generator():
            for sample in generator:
                sample = self.preprocess_sample(sample)
                for comp in sample[0]:
                    yield comp
                for comp in sample[1]:
                    yield comp

        self.sub_tokenizer.build_vocabulary(
            generator=sub_comp_generator(),
            add_start=add_start,
            add_eos=add_eos,
            add_pad=add_pad,
            **kwargs,
        )

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_num_sub_exprs(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "num_sub_exprs" in config:
                config["num_sub_seqs"] = config.pop("num_sub_exprs")

        return [preprocess_num_sub_exprs] + super().config_preprocessors()
