"""Decomposed LTL specification tokenizer"""

from copy import deepcopy
from typing import Any, Dict, Type

from numpy.random import default_rng

from ...registry import register_type
from ...tokenizers.decomp_expr_pair_tokenizers import DecompExprPairToDecompSeqTPETokenizer
from ...tokenizers.to_decomp_seq_pos_tokenizer import ToDecompSeqPosTokenizer
from ...tokenizers.tokenizer import TokenizationException
from .decomp_ltl_spec import DecompLTLSpec
from .ltl_spec import LTLSpec

DEFAULT_ASSUMPTION_TOKEN = "<a>"


@register_type
class DecompLTLSpecToSeqTPETokenizer(
    DecompExprPairToDecompSeqTPETokenizer[LTLSpec, DecompLTLSpec]
):
    def __init__(
        self,
        num_props: int,
        prop_tokenizer: ToDecompSeqPosTokenizer[LTLSpec, DecompLTLSpec],
        dtype: Type[DecompLTLSpec] = DecompLTLSpec,
        rename_aps_random: bool = False,
        rename_aps: bool = False,
        num_inputs: int = None,
        num_outputs: int = None,
        **kwargs,
    ):
        self.rename_aps_random = rename_aps_random
        self.rename_aps_order = rename_aps
        assert not (rename_aps and rename_aps_random)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        super().__init__(
            dtype=dtype, sub_tokenizer=prop_tokenizer, num_sub_seqs=num_props, **kwargs
        )

    def preprocess_sample(self, x: DecompLTLSpec) -> DecompLTLSpec:
        if self.rename_aps_random:
            if self.num_outputs is not None and self.num_inputs is not None:
                inputs_vocab = ["i" + str(i) for i in range(self.num_inputs)]
                outputs_vocab = ["o" + str(i) for i in range(self.num_outputs)]
            else:
                inputs_vocab = list(
                    filter(lambda x: x.startswith("i"), self.vocabulary.token_to_id)
                )
                outputs_vocab = list(
                    filter(lambda x: x.startswith("o"), self.vocabulary.token_to_id)
                )
            if len(x.inputs) > len(inputs_vocab):
                raise TokenizationException("Too many input variables")
            if len(x.outputs) > len(outputs_vocab):
                raise TokenizationException("Too many input variables")
            rng = default_rng()
            new_i = sorted(list(rng.choice(inputs_vocab, len(x.inputs), replace=False)))
            new_o = sorted(list(rng.choice(outputs_vocab, len(x.outputs), replace=False)))
            rename = {k: v for k, v in zip(x.inputs + x.outputs, new_i + new_o)}
            dx = deepcopy(x)
            dx.rename_aps(renaming=rename)
            return dx
        elif self.rename_aps_order:
            dx = deepcopy(x)
            dx.rename_aps(random=False)
            return dx
        else:
            return x

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_num_props(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "num_sub_exprs" in config:
                config["num_props"] = config.pop("num_sub_exprs")

        def postprocess_prop_tokenizer(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            if "sub_tokenizer" in config:
                config["prop_tokenizer"] = config.pop("sub_tokenizer")

        return super().config_postprocessors() + [
            postprocess_num_props,
            postprocess_prop_tokenizer,
        ]

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_prop_tokenizer(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            prop_tokenizer_keys = ["prop_pad", "pos_pad", "notation", "tpe_format"]
            if "prop_tokenizer" in config:
                if isinstance(config["prop_tokenizer"], str) and any(
                    k in config for k in prop_tokenizer_keys
                ):
                    config["prop_tokenizer"] = {"base": config["prop_tokenizer"]}

                for k in prop_tokenizer_keys:
                    if k in config:
                        if k == "prop_pad":
                            config["prop_tokenizer"]["pad"] = config.pop(k)
                        else:
                            config["prop_tokenizer"][k] = config.pop(k)
            else:
                config["prop_tokenizer"] = {
                    "type": "LTLSpecToSeqTPETokenizer",
                    "pad": config.pop("prop_pad"),
                    "pos_pad": config.pop("pos_pad"),
                    "notation": config.pop("notation", "prefix"),
                    "tpe_format": config.pop("tpe_format", "branch-up"),
                    "default_name": "sub-tokenizer",
                    "project": config["project"],
                }

        def preprocess_num_sub_seqs(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("num_sub_seqs", None)
            annotations.pop("num_sub_seqs", None)

        return [
            preprocess_prop_tokenizer,
            preprocess_num_sub_seqs,
        ] + super().config_preprocessors()
