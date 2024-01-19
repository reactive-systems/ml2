"""Tokenizer that encodes a decomposed data type into a decomposed sequence encoding with positional encoding"""


from typing import Generator, Generic, List, TypeVar

from ...dtypes import DecompDType, DType
from ...registry import register_type
from ..to_decomp_seq_pos_tokenizer import DecompSeqPosEncoding, ToDecompSeqPosTokenizer
from ..to_seq_pos_tokenizer import SeqPosEncoding
from ..tokenizer import PAD_TOKEN

S = TypeVar("S", bound=DType)
DT = DecompDType[S]
T = TypeVar("T", bound=DT)


@register_type
class DecompDTypeToDecompSeqPosTokenizer(ToDecompSeqPosTokenizer[S, T], Generic[S, T]):
    def encode(self, data: T, **kwargs) -> DecompSeqPosEncoding:
        if len(data) > self.num_sub_seqs:
            raise Exception("Too many components")
        encs = []
        for comp in data:
            encs.append(self.sub_tokenizer.encode(comp, **kwargs))
        for _ in range(len(data), self.num_sub_seqs):
            encs.append(
                SeqPosEncoding(
                    tokens=[],
                    pad_tokens=[PAD_TOKEN] * self.sub_tokenizer.pad,
                    ids=self.vocabulary.tokens_to_ids([PAD_TOKEN] * self.sub_tokenizer.pad),
                    pos_enc=[[] * self.sub_tokenizer.pad],
                    pad_pos_enc=[[0] * self.sub_tokenizer.pos_pad] * self.sub_tokenizer.pad,
                )
            )
        return DecompSeqPosEncoding(seq_pos_encs=encs)

    def decode(self, decomp_ids: List[List[int]], **kwargs) -> T:
        components = [self.sub_tokenizer.decode(ids, **kwargs) for ids in decomp_ids]
        return self.dtype.from_components(components=components, **kwargs)

    def build_vocabulary(
        self,
        generator: Generator[T, None, None],
        add_start: bool = False,
        add_eos: bool = False,
        add_pad: bool = False,
        **kwargs,
    ) -> None:
        def sub_comp_generator():
            for sample in generator:
                for comp in sample:
                    yield comp

        self.sub_tokenizer.build_vocabulary(
            generator=sub_comp_generator(),
            add_start=add_start,
            add_eos=add_eos,
            add_pad=add_pad,
            **kwargs,
        )
