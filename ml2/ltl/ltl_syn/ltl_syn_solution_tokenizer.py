"""LTL synthesis solution tokenizer"""

from typing import Any, Dict, List, Type

from ...aiger import AIGERCircuit, AIGERToSeqTokenizer
from ...dtypes.pair import Pair
from ...registry import register_type
from ...tokenizers import CatSeqPairToSeqTokenizer, CatToIdTokenizer, Vocabulary
from .ltl_syn_problem import LTLSynSolution
from .ltl_syn_status import LTLSynStatus

NEWLINE_TOKEN = "<n>"
COMPLEMENT_TOKEN = "<c>"
LATCH_TOKEN = "<l>"
REALIZABLE_TOKEN = "<r>"
UNREALIZABLE_TOKEN = "<u>"


@register_type
class LTLSynSolutionToSeqTokenizer(
    CatSeqPairToSeqTokenizer[LTLSynStatus, AIGERCircuit, LTLSynSolution]
):
    def __init__(
        self,
        pad: int,
        components: List[str] = None,
        dtype: Type[LTLSynSolution] = LTLSynSolution,
        eos: bool = False,
        inputs: List[str] = None,
        outputs: List[str] = None,
        unfold_negations: bool = False,
        unfold_latches: bool = False,
        ints_as_chars: bool = False,
        start: bool = False,
        swap: bool = False,
        vocabulary: Vocabulary = None,
        name: str = "tokenizer",
        project: str = None,
        **kwargs
    ) -> None:
        """
        inputs, outputs: only required if components does not contain header or symbols
        """

        cat_tokenizer = CatToIdTokenizer(
            dtype=LTLSynStatus,
            vocabulary=None,
            name=name + "/status-tokenizer",
            project=project,
            **kwargs,
        )

        seq_tokenizer = AIGERToSeqTokenizer(
            pad=None,
            components=components,
            eos=None,
            inputs=inputs,
            outputs=outputs,
            unfold_negations=unfold_negations,
            unfold_latches=unfold_latches,
            ints_as_chars=ints_as_chars,
            start=None,
            vocabulary=None,
            name=name + "/aiger-tokenizer",
            project=project,
            **kwargs,
        )

        super().__init__(
            dtype=dtype,
            cat_tokenizer=cat_tokenizer,
            seq_tokenizer=seq_tokenizer,
            pad=pad,
            eos=eos,
            start=start,
            swap=swap,
            vocabulary=vocabulary,
            name=name,
            project=project,
            **kwargs,
        )

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_child_tokenizers(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            config.pop("cat_tokenizer", None)
            annotations.pop("cat_tokenizer", None)
            config.pop("seq_tokenizer", None)
            annotations.pop("seq_tokenizer", None)
            config["inputs"] = self.seq_tokenizer.inputs
            config["outputs"] = self.seq_tokenizer.outputs
            config["components"] = self.seq_tokenizer.components
            config["unfold_negations"] = self.seq_tokenizer.unfold_negations
            config["unfold_latches"] = self.seq_tokenizer.unfold_latches
            config["ints_as_chars"] = self.seq_tokenizer.ints_as_chars

        return [postprocess_child_tokenizers] + super().config_postprocessors()

    def decode_tokens(self, ids: List[str], **kwargs) -> Pair[LTLSynStatus, AIGERCircuit]:
        if self.swap:
            status_token = ids[-1]
            circuit_token = ids[:-1]
        else:
            status_token = ids[0]
            circuit_token = ids[1:]
        status = self.cat_tokenizer.decode_token(status_token, **kwargs)
        circuit = self.seq_tokenizer.decode_tokens(
            circuit_token, realizable=status.realizable, **kwargs
        )
        return self.dtype.from_components(fst=status, snd=circuit, **kwargs)
