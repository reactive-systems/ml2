"""Propositional satisfiability experiment"""

import logging
from typing import List

import numpy as np
import tensorflow as tf

from ..seq2seq_experiment import Seq2SeqExperiment
from ..tools.limboole import Limboole
from .assignment_encoder import AssignmentEncoder
from .prop_formula import PropFormula
from .prop_sat_data import PropSatSplitData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropSatExperiment(Seq2SeqExperiment):

    BUCKET_DIR = "prop-sat"
    WANDB_PROJECT = "prop-sat"

    def __init__(
        self,
        batch_size: int = 256,
        dataset_name: str = "rfa-0",
        props: List[str] = None,
        max_input_length: int = 128,
        max_target_length: int = 128,
        **kwargs,
    ):
        self.props = props if props else ["a", "b", "c", "d", "e"]
        super().__init__(
            batch_size=batch_size,
            dataset_name=dataset_name,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            **kwargs,
        )

    @property
    def init_verifier(self):
        return Limboole()

    @property
    def init_dataset(self):
        return PropSatSplitData.load(self.dataset_name)

    @property
    def init_target_encoder(self):
        return AssignmentEncoder(
            start=True, eos=True, pad=self.max_target_length, encode_start=False
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
            beam_result["assignment"] = self.target_encoder.assignment
            if verify:
                # pylint: disable=E1102

                beam_result["verification"] = self.verifier.check_solution(
                    formula, beam_result["assignment"]
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
        parser.add_argument("--props", nargs="*", default=defaults["props"])


if __name__ == "__main__":
    e = PropSatExperiment()
