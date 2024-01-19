"""TensorFlow Transformer pipeline"""

import copy
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..configurable import Configurable
from ..datasets import Dataset
from ..dtypes import DType, Supervised
from ..models import tf_transformer
from ..registry import register_type
from ..utils.tf_utils import (
    str_to_tf_float_dtype,
    str_to_tf_int_dtype,
    tf_float_dtype_to_str,
    tf_int_dtype_to_str,
)
from .callbacks import Callback
from .metrics import (
    Acc,
    AccPerSeq,
    Counter,
    EvalErrCounter,
    EvalSupervisedErrCounter,
    Metric,
    MetricGroup,
    NullMetric,
)
from .samples import Beam, BeamSearchLabeledSample, BeamSearchSample
from .seq2seq_pipeline import Seq2SeqPipeline
from .tf_sl_pipeline import TFSLPipeline

I = TypeVar("I", bound=DType)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
@register_type
class TFTransformerConfig(Configurable):
    alpha: float = 0.5
    beam_size: int = 2
    d_embed: int = 256
    d_embed_dec: int = None
    d_embed_enc: int = None
    d_ff: int = 1024
    d_ff_dec: int = None
    d_ff_enc: int = None
    dropout: float = 0.0
    dtype_float: tf.DType = tf.float32
    dtype_int: tf.DType = tf.int32
    ff_activation: str = "relu"
    num_heads: int = 4
    num_heads_dec: int = None
    num_heads_enc: int = None
    num_layers: int = 8
    num_layers_dec: int = None
    num_layers_enc: int = None

    def __post_init__(self):
        if self.d_embed is not None:
            if self.d_embed_enc is None:
                self.d_embed_enc = self.d_embed
            if self.d_embed_dec is None:
                self.d_embed_dec = self.d_embed
        if self.d_ff is not None:
            if self.d_ff_enc is None:
                self.d_ff_enc = self.d_ff
            if self.d_ff_dec is None:
                self.d_ff_dec = self.d_ff
        if self.num_heads is not None:
            if self.num_heads_enc is None:
                self.num_heads_enc = self.num_heads
            if self.num_heads_dec is None:
                self.num_heads_dec = self.num_heads
        if self.num_layers is not None:
            if self.num_layers_enc is None:
                self.num_layers_enc = self.num_layers
            if self.num_layers_dec is None:
                self.num_layers_dec = self.num_layers
        if self.d_embed_enc % self.num_heads_enc != 0:
            raise ValueError(
                f"Encoder embedding dimension {self.d_embed_enc} is not divisible by the number of encoder attention heads {self.num_heads_enc}"
            )
        if self.d_embed_dec % self.num_heads_dec != 0:
            raise ValueError(
                f"Decoder embedding dimension {self.d_embed_dec} is not divisible by the number of decoder attention heads {self.num_heads_dec}"
            )

    def asdict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["d_embed", "d_ff", "num_heads", "num_layers"]
        }

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_abbreviations(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            for abb in ["d_embed", "d_ff", "num_heads", "num_layers"]:
                config.pop(abb, None)
                annotations.pop(abb, None)

        def postprocess_tf_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "dtype_float"
            if name in config and isinstance(config[name], tf.DType):
                config[name] = tf_float_dtype_to_str[config[name]]

        def postprocess_tf_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "dtype_int"
            if name in config and isinstance(config[name], tf.DType):
                config[name] = tf_int_dtype_to_str[config[name]]

        return [
            postprocess_abbreviations,
            postprocess_tf_float_dtype,
            postprocess_tf_int_dtype,
        ] + super().config_postprocessors()

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_tf_float_dtype(
            config: Dict[str, Any], annotations: Dict[str, type]
        ) -> None:
            name = "dtype_float"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_tf_float_dtype[config[name]]

        def preprocess_tf_int_dtype(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            name = "dtype_int"
            if name in config and isinstance(config[name], str):
                config[name] = str_to_tf_int_dtype[config[name]]

        return [
            preprocess_tf_float_dtype,
            preprocess_tf_int_dtype,
        ] + super().config_preprocessors()


@register_type
class TFTransformerPipeline(Seq2SeqPipeline[I, T], TFSLPipeline[I, T]):
    def __init__(
        self,
        model_config: TFTransformerConfig,
        attn_mask_1d: bool = False,
        attn_mask_2d: bool = False,
        attn_mask_3d: bool = False,
        attn_mask_4d: bool = False,
        custom_pos_enc: bool = False,
        **kwargs,
    ) -> None:
        self.attn_mask_1d = attn_mask_1d
        self.attn_mask_2d = attn_mask_2d
        self.attn_mask_3d = attn_mask_3d
        self.attn_mask_4d = attn_mask_4d
        self.custom_pos_enc = custom_pos_enc
        super().__init__(model_config=model_config, **kwargs)
        self._attn_model = None

    @property
    def attn_model(self):
        if not self._attn_model:
            self._attn_model = self.init_model(training=False, attn_weights=True)
            logger.info("Created attention model")
            checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            if checkpoint:
                logger.info("Found checkpoint %s", checkpoint)
                self._attn_model.load_weights(checkpoint).expect_partial()
                logger.info("Loaded weights from checkpoint")
        return self._attn_model

    def init_model(
        self,
        training: bool = False,
        attn_weights: bool = False,
        **kwargs,
    ):
        config = asdict(self.model_config)
        config["input_vocab_size"] = self.input_vocab_size
        config["input_eos_id"] = self.input_eos_id
        config["input_pad_id"] = self.input_pad_id
        config["target_vocab_size"] = self.target_vocab_size
        config["target_start_id"] = self.target_start_id
        config["target_eos_id"] = self.target_eos_id
        config["target_pad_id"] = self.target_pad_id
        config["max_encode_length"] = self.max_input_length
        config["max_decode_length"] = self.max_target_length
        config["num_replica"] = getattr(self, "num_replica", None)
        return tf_transformer.create_model(
            config,
            training=training,
            input_attn_mask_1d=self.attn_mask_1d,
            input_attn_mask_2d=self.attn_mask_2d,
            input_attn_mask_3d=self.attn_mask_3d,
            input_attn_mask_4d=self.attn_mask_4d,
            custom_pos_enc=self.custom_pos_enc,
            attn_weights=attn_weights,
        )

    def __call__(self, x: I, training: bool = False, **kwargs) -> T:
        return self.eval_sample(x).pred

    def decode(self, prediction_encoding, input: Optional[I] = None) -> T:
        return self.target_tokenizer.decode(prediction_encoding)

    def build_vocabulary(self, dataset: Dataset[Supervised[I, T]]) -> None:
        def input_generator():
            for x in dataset.generator():
                yield x.input

        def target_generator():
            for x in dataset.generator():
                yield x.target

        logger.info(f"Building input vocabulary with dataset {dataset.name}")
        self.input_tokenizer.build_vocabulary(input_generator())
        logger.info(f"Building target vocabulary with dataset {dataset.name} and start token")
        self.target_tokenizer.build_vocabulary(target_generator(), add_start=True)

    def convert_sl_dataset_to_tf(
        self, dataset: Dataset[Supervised[I, T]], return_error_callbacks: bool = False
    ) -> tf.data.Dataset:
        def shape_dataset(input_tensor, target_tensor):
            if type(input_tensor) is tuple:
                return ((*input_tensor, target_tensor), target_tensor)
            return ((input_tensor, target_tensor), target_tensor)

        if return_error_callbacks:
            tf_dataset, err_callbacks = super().convert_sl_dataset_to_tf(
                dataset=dataset, return_error_callbacks=True
            )
            return tf_dataset.map(shape_dataset), err_callbacks
        else:
            tf_dataset = super().convert_sl_dataset_to_tf(
                dataset=dataset, return_error_callbacks=False
            )
            return tf_dataset.map(shape_dataset)

    def eval_attn_weights(self, x: I, training: bool = False):
        attn = {}

        encoding = self.input_tokenizer.encode(x)
        enc_tensor = self.input_tokenizer.encode_tf(x)
        if self.custom_pos_enc:
            enc_tensor, pos_enc_tensor = enc_tensor
            # pylint: disable=E1102
            preds, _, enc_attn, dec_attn = self.attn_model(
                (tf.expand_dims(enc_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
                training=training,
            )  # [0]
        else:
            preds, _, enc_attn, dec_attn = self.attn_model(
                tf.expand_dims(enc_tensor, axis=0), training=training
            )[0]

        num_input_tokens = len(encoding.tokens)
        enc_self_attn_dict_list = []
        # iterate over attention heads
        for head in range(self.model_config.num_heads):
            layerdict = {}
            for layer in range(1, self.model_config.num_layers_enc + 1):
                playerdict = {}
                for player in range(num_input_tokens):
                    attended_player_dict = {}
                    for player_attended in range(num_input_tokens):
                        att = enc_attn[f"layer_{layer}"]["self_attn"][0][head][player][
                            player_attended
                        ].numpy()
                        # In layer *layer* player *player* attends to player *player attended* by att
                        attended_player_dict[player_attended] = str(att)
                    playerdict[player] = attended_player_dict
                layerdict[layer] = playerdict
            enc_self_attn_dict_list.append(layerdict)

        attn["enc_attn"] = enc_self_attn_dict_list
        attn["dec_attn"] = []
        attn["enc_dec_attn"] = []

        pred_tokens = []
        for beam in preds[0]:
            # result = self.target_tokenizer.decode_tf(beam)
            pad_pred_tokens = self.target_tokenizer.vocabulary.ids_to_tokens(beam.numpy().tolist())
            pred_tokens.append(pad_pred_tokens)
            num_pred_tokens = len(pad_pred_tokens)

            dec_self_attn_dict_list = []
            enc_dec_attn_dict_list = []
            for head in range(self.model_config.num_heads):
                dec_self_attn_layer_dict = {}
                enc_dec_attn_layer_dict = {}

                for layer in range(1, self.model_config.num_layers_dec + 1):
                    dec_self_attn_player_dict = {}
                    enc_dec_attn_player_dict = {}

                    for player in range(num_pred_tokens):
                        attended_dec_player_dict = {}
                        for player_attended in range(num_pred_tokens):
                            att = dec_attn[f"layer_{layer}"]["self_attn"][0][head][player][
                                player_attended
                            ].numpy()
                            attended_dec_player_dict[player_attended] = str(att)
                        dec_self_attn_player_dict[player] = attended_dec_player_dict
                        attended_enc_player_dict = {}
                        for player_attended in range(0, num_input_tokens):
                            att = dec_attn[f"layer_{layer}"]["enc_dec_attn"][0][head][player][
                                player_attended
                            ].numpy()
                            attended_enc_player_dict[player_attended] = str(att)
                        enc_dec_attn_player_dict[player] = attended_enc_player_dict

                    dec_self_attn_layer_dict[layer] = dec_self_attn_player_dict
                    enc_dec_attn_layer_dict[layer] = enc_dec_attn_player_dict

                dec_self_attn_dict_list.append(dec_self_attn_layer_dict)
                enc_dec_attn_dict_list.append(enc_dec_attn_layer_dict)

            attn["dec_attn"].append(dec_self_attn_dict_list)
            attn["enc_dec_attn"].append(enc_dec_attn_dict_list)

        return (encoding.tokens, pred_tokens, attn)

    def eval(
        self,
        dataset: Dataset[I],
        batch_size: int = 32,
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        **kwargs,
    ) -> Generator[BeamSearchSample, None, None]:
        if metric is None:
            metric = NullMetric()
        if callbacks is None:
            callbacks = []

        logger.info(f"Evaluating dataset {dataset.name}")
        pbar = tqdm(desc="Evaluated samples", unit="sample")

        samples = []
        errors = []

        def tensor_generator():
            for sample in self.init_samples(dataset=dataset, return_errors=True):
                if sample.inp_enc is None:
                    errors.append(sample)
                    continue

                samples.append(sample)
                if self.custom_pos_enc:
                    yield (sample.inp_enc[0], sample.inp_enc[1])
                else:
                    yield sample.inp_enc

        tf_dataset = tf.data.Dataset.from_generator(
            tensor_generator,
            output_signature=self.input_tokenizer.tf_signature,
        )

        for batch in tf_dataset.batch(batch_size, drop_remainder=False):
            eval_samples = []
            start = time.time()
            predictions, _ = self.eval_model(batch, training=False)  # pylint: disable=E1102
            for pred in predictions:
                sample = samples.pop(0)
                for beam_id, beam in enumerate(pred):
                    try:
                        # TODO start id hack
                        pred = self.decode(
                            prediction_encoding=np.array(beam)[1:], input=sample.inp
                        )
                        end = time.time()
                        sample.add_beam(
                            Beam(
                                id=beam_id,
                                pred=pred,
                                pred_enc=np.array(beam)[1:],
                                time=end - start,
                            )
                        )
                    except Exception as err:
                        end = time.time()
                        sample.add_beam(
                            Beam(
                                id=beam_id,
                                pred_enc=np.array(beam)[1:],
                                pred_dec_err=str(err),
                                time=end - start,
                            )
                        )

                if sample.pred_dec_err is None:
                    eval_samples.append(sample)
                else:
                    errors.append(sample)

            for sample in errors + eval_samples:
                metric.add(sample)
                [callback.add(sample) for callback in callbacks]
                pbar.update()
                pbar.set_postfix(metric.compute_dict())
            for sample in errors + eval_samples if return_errors else eval_samples:
                yield sample

            errors = []

        for err in errors:
            metric.add(err)
            [callback.add(err) for callback in callbacks]
            pbar.update()
            pbar.set_postfix(metric.compute_dict())
            if return_errors:
                yield err

        pbar.close()

    def eval_sample(self, x: I, training: bool = False, **kwargs) -> BeamSearchSample:
        sample = self.init_sample(x)

        if sample.inp_enc is None:
            return sample

        start = time.time()
        if self.custom_pos_enc:
            formula_tensor, pos_enc_tensor = sample.inp_enc
            # pylint: disable=E1102

            preds = self.eval_model(
                (tf.expand_dims(formula_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
                training=training,
            )[0]
        else:
            preds = self.eval_model(tf.expand_dims(sample.inp_enc, axis=0), training=training)[0]

        for beam_id, beam in enumerate(preds[0]):
            # TODO start id hack
            pred_enc = np.array(beam)[1:]
            try:
                pred = self.decode(prediction_encoding=pred_enc, input=sample.inp)
                end = time.time()
                sample.add_beam(Beam(id=beam_id, pred=pred, pred_enc=pred_enc, time=end - start))
            except Exception as err:
                end = time.time()
                sample.add_beam(
                    Beam(id=beam_id, pred_enc=pred_enc, pred_dec_err=str(err), time=start - end)
                )
        return sample

    def eval_supervised(
        self,
        dataset: Dataset[Supervised[I, T]],
        batch_size: int = 32,
        metric: Metric = None,
        callbacks: List[Callback] = None,
        return_errors: bool = True,
        **kwargs,
    ) -> Generator[BeamSearchLabeledSample, None, None]:
        if metric is None:
            metric = NullMetric()
        if callbacks is None:
            callbacks = []

        logger.info(f"Evaluating dataset {dataset.name}")
        pbar = tqdm(desc="Evaluated samples", unit="sample")

        samples = []
        errors = []

        def tensor_generator():
            for sample in self.init_supervised_samples(dataset=dataset, return_errors=True):
                if sample.inp_enc is None or sample.tar_enc is None:
                    errors.append(sample)
                    continue

                samples.append(sample)
                if self.custom_pos_enc:
                    yield (sample.inp_enc[0], sample.inp_enc[1])
                else:
                    yield sample.inp_enc

        tf_dataset = tf.data.Dataset.from_generator(
            tensor_generator,
            output_signature=self.input_tokenizer.tf_signature,
        )

        for batch in tf_dataset.batch(batch_size, drop_remainder=False):
            eval_samples = []
            start = time.time()
            predictions, _ = self.eval_model(batch, training=False)  # pylint: disable=E1102
            for pred in predictions:
                sample = samples.pop(0)
                for beam_id, beam in enumerate(pred):
                    try:
                        # TODO start id hack
                        pred = self.decode(
                            prediction_encoding=np.array(beam)[1:], input=sample.inp
                        )
                        end = time.time()
                        sample.add_beam(
                            Beam(
                                id=beam_id,
                                pred=pred,
                                pred_enc=np.array(beam)[1:],
                                time=end - start,
                            )
                        )
                    except Exception as err:
                        end = time.time()
                        sample.add_beam(
                            Beam(
                                id=beam_id,
                                pred_enc=np.array(beam)[1:],
                                pred_dec_err=str(err),
                                time=end - start,
                            )
                        )

                if sample.pred_dec_err is None:
                    eval_samples.append(sample)
                else:
                    errors.append(sample)

            for sample in errors + eval_samples:
                metric.add(sample)
                [callback.add(sample) for callback in callbacks]
                pbar.update()
                pbar.set_postfix(metric.compute_dict())
            for sample in errors + eval_samples if return_errors else eval_samples:
                yield sample

            errors = []

        for err in errors:
            metric.add(err)
            [callback.add(err) for callback in callbacks]
            pbar.update()
            pbar.set_postfix(metric.compute_dict())
            if return_errors:
                yield err

        pbar.close()

    def eval_supervised_sample(self, x: Supervised[I, T], **kwargs) -> BeamSearchLabeledSample:
        raise NotImplementedError()

    def init_sample(self, x: I) -> BeamSearchSample:
        return BeamSearchSample(**asdict(super().init_sample(x)))

    def init_samples(
        self, dataset: Dataset[I], return_errors: bool = False
    ) -> Generator[BeamSearchSample, None, None]:
        for sample in super().init_samples(dataset=dataset, return_errors=return_errors):
            yield BeamSearchSample(**asdict(sample))

    def init_supervised_samples(
        self, dataset: Dataset[Supervised[I, T]], return_errors: bool = False
    ) -> Generator[BeamSearchLabeledSample, None, None]:
        for sample in super().init_supervised_samples(
            dataset=dataset, return_errors=return_errors
        ):
            yield BeamSearchLabeledSample(**asdict(sample))

    @staticmethod
    def default_metric() -> Metric:
        return MetricGroup([Counter(), EvalErrCounter()])

    @staticmethod
    def default_supervised_metric() -> Metric:
        return MetricGroup(
            [
                Acc(pad_same_length=True),
                AccPerSeq(pad_same_length=True),
                Counter(),
                EvalSupervisedErrCounter(),
            ]
        )

    @staticmethod
    def expand_eval_config(config: dict) -> List[Tuple[dict, dict]]:
        expanded_configs = [(defaultdict(dict), config)]

        # expand lists of alphas to multiple configs
        alpha_expanded_configs = []
        for delimiters, config in expanded_configs:
            pipe_config = config["pipeline"]
            if "alpha" in pipe_config and isinstance(pipe_config["alpha"], list):
                for alpha in pipe_config["alpha"]:
                    expanded_config = copy.deepcopy(config)
                    expanded_config["pipeline"]["alpha"] = alpha
                    expanded_delimiters = copy.deepcopy(delimiters)
                    expanded_delimiters["pipeline"]["alpha"] = alpha
                    alpha_expanded_configs.append((expanded_delimiters, expanded_config))
            else:
                alpha_expanded_configs.append((delimiters, config))

        # expand lists of beam sizes to multiple configs
        bs_expanded_configs = []
        for delimiters, config in alpha_expanded_configs:
            pipe_config = config["pipeline"]
            if "beam_size" in pipe_config and isinstance(pipe_config["beam_size"], list):
                for bs in pipe_config["beam_size"]:
                    expanded_config = copy.deepcopy(config)
                    expanded_config["pipeline"]["beam_size"] = bs
                    expanded_delimiters = copy.deepcopy(delimiters)
                    expanded_delimiters["pipeline"]["beam_size"] = bs
                    bs_expanded_configs.append((expanded_delimiters, expanded_config))
            else:
                bs_expanded_configs.append((delimiters, config))

        # move alpha and beam size into model_config dict
        model_expanded_configs = []
        for delimiters, config in bs_expanded_configs:
            pipe_config = config["pipeline"]
            model_config = pipe_config.get("model_config", {})
            if "alpha" in pipe_config:
                model_config["alpha"] = pipe_config.pop("alpha")
            if "beam_size" in pipe_config:
                model_config["beam_size"] = pipe_config.pop("beam_size")
            pipe_config["model_config"] = model_config
            model_expanded_configs.append((dict(delimiters), config))

        return model_expanded_configs
