"""Prototypical class for sequence to sequence experiments"""

import logging
import os

from tensorflow import keras

from .experiment import Experiment

logger = logging.getLogger(__name__)


class EncoderErrorCallback(keras.callbacks.Callback):
    def __init__(self, dataset, model_dir):
        super().__init__()
        self.dataset = dataset
        self.model_dir = model_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            return
        input_errors = self.dataset.input_encoder_errors
        if any([input_errors[split] for split in self.dataset.split_names]):
            logger.info("When encoding inputs errors occured: %s", input_errors)
            input_errors_filepath = os.path.join(self.model_dir, "input-encoder.err")
            with open(input_errors_filepath, "w") as input_errors_file:
                for split in self.dataset.split_names:
                    if input_errors[split]:
                        input_errors_file.write(split.upper() + "\n")
                        for err, count in input_errors[split].items():
                            input_errors_file.write(f"{err}: {count}\n")
        target_errors = self.dataset.target_encoder_errors
        if any([target_errors[split] for split in self.dataset.split_names]):
            logger.info("When encoding targets errors occurred: %s", target_errors)
            target_errors_filepath = os.path.join(self.model_dir, "target-encoder.err")
            with open(target_errors_filepath, "w") as target_errors_file:
                for split in self.dataset.split_names:
                    if target_errors[split]:
                        target_errors_file.write(split.upper() + "\n")
                        for err, count in target_errors[split].items():
                            target_errors_file.write(f"{err}: {count}\n")


class Seq2SeqExperiment(Experiment):
    def __init__(
        self,
        alpha: float = 0.5,
        beam_size: int = 1,
        checkpoint_monitor: str = "val_accuracy_per_sequence",
        max_input_length: int = 32,
        max_target_length: int = 32,
        **kwargs,
    ):
        self.alpha = alpha
        self.beam_size = beam_size
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self._input_encoder = None
        self._target_encoder = None
        super().__init__(checkpoint_monitor=checkpoint_monitor, **kwargs)

    @property
    def callbacks(self):
        return super().callbacks + [EncoderErrorCallback(self.dataset, self.local_dir)]

    @property
    def target_encoder(self):
        if not self._target_encoder:
            self._target_encoder = self.init_target_encoder
            if not self._target_encoder.load_vocabulary(self.local_dir):
                logger.info("Building target encoder vocabulary")
                self._target_encoder.build_vocabulary(self.dataset.target_generator())
                self._target_encoder.vocabulary_to_file(self.local_dir)
            logger.info("Initialized target encoder")
        return self._target_encoder

    @property
    def input_encoder(self):
        if not self._input_encoder:
            self._input_encoder = self.init_input_encoder
            if not self._input_encoder.load_vocabulary(self.local_dir):
                logger.info("Building input encoder vocabulary")
                self._input_encoder.build_vocabulary(self.dataset.input_generator())
                self._input_encoder.vocabulary_to_file(self.local_dir)
            logger.info("Initialized input encoder")
        return self._input_encoder

    @property
    def init_input_encoder(self):
        raise NotImplementedError

    @property
    def init_target_encoder(self):
        raise NotImplementedError

    @property
    def init_tf_dataset(self):
        return self.dataset.tf_dataset(self.input_encoder, self.target_encoder)

    @property
    def input_vocab_size(self):
        return self.input_encoder.vocabulary.size()

    @property
    def input_pad_id(self):
        return self.input_encoder.vocabulary.token_to_id.get("<p>", None)

    @property
    def input_eos_id(self):
        return self.input_encoder.vocabulary.token_to_id.get("<e>", None)

    @property
    def target_vocab_size(self):
        return self.target_encoder.vocabulary.size()

    @property
    def target_start_id(self):
        return self.target_encoder.vocabulary.token_to_id.get("<s>", None)

    @property
    def target_eos_id(self):
        return self.target_encoder.vocabulary.token_to_id.get("<e>", None)

    @property
    def target_pad_id(self):
        return self.target_encoder.vocabulary.token_to_id.get("<p>", None)

    @classmethod
    def add_init_args(cls, parser):
        super().add_init_args(parser)
        defaults = cls.get_default_args()
        parser.add_argument("--alpha", type=float, default=defaults["alpha"])
        parser.add_argument("--beam-size", type=int, default=defaults["beam_size"])
        parser.add_argument("--max-input-length", type=int, default=defaults["max_input_length"])
        parser.add_argument("--max-target-length", type=int, default=defaults["max_target_length"])

    @classmethod
    def add_eval_args(cls, parser):
        super().add_eval_args(parser)
        parser.add_argument("--alphas", nargs="*", type=int, default=[0.5])
        parser.add_argument("--beam-sizes", nargs="*", type=int, default=[1, 4, 8, 16])
        parser.add_argument("--samples", type=int, default=1024)
