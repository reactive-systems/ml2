"""LTL satisfiability experiment"""

import logging
import numpy as np
import tensorflow as tf

from ...tools.spot import Spot
from ...trace import SymbolicTraceEncoder
from ...seq2seq_experiment import Seq2SeqExperiment
from .ltl_sat_data import LTLSatSplitData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSatExperiment(Seq2SeqExperiment):

    BUCKET_DIR = "ltl-sat"
    WANDB_PROJECT = "ltl-sat"

    def __init__(
        self,
        batch_size: int = 256,
        dataset_name: str = "rft-0",
        aps: list = None,
        max_input_length: int = 128,
        max_target_length: int = 128,
        **kwargs,
    ):
        self.aps = aps if aps else ["a", "b", "c", "d", "e"]
        super().__init__(
            batch_size=batch_size,
            dataset_name=dataset_name,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            **kwargs,
        )

    @property
    def init_verifier(self):
        return Spot()

    @property
    def init_dataset(self):
        return LTLSatSplitData.load(self.dataset_name)

    @property
    def init_target_encoder(self):
        return SymbolicTraceEncoder(
            notation="infix",
            encoded_notation="prefix",
            start=True,
            eos=True,
            pad=self.max_target_length,
            encode_start=False,
        )

    def call(self, formula: str, training: bool = False, verify: bool = False):
        if not self.input_encoder.encode(formula):
            logger.info("Econding error: %s", self.input_encoder.error)
            return None
        formula_tensor, pos_enc_tensor = self.input_encoder.tensor
        # pylint: disable=E1102
        preds = self.eval_model(
            (tf.expand_dims(formula_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
            training=training,
        )[0]
        results = []
        for beam in preds[0]:
            if not self.target_encoder.decode(np.array(beam)):
                logger.info("Decoding error: %s", self.target_encoder.error)
                # return None
            beam_result = {}
            beam_result["trace"] = self.target_encoder.sequence
            if verify:
                # pylint: disable=E1102
                beam_result["verification"] = self.verifier.mc_trace(
                    formula, beam_result["trace"].replace("{", "cycle{")
                )
            results.append(beam_result)
        return results

    @classmethod
    def add_eval_args(cls, parser):
        super().add_eval_args(parser)
        parser.add_argument("-d", "--data", nargs="*", default=None)

    @classmethod
    def add_init_args(cls, parser):
        super().add_init_args(parser)
        defaults = cls.get_default_args()
        parser.add_argument("--aps", nargs="*", default=defaults["aps"])
