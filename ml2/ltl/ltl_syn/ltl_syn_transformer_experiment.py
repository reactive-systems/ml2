"""LTL synthesis using the Transformer"""

import logging
import sys
import numpy as np
import tensorflow as tf

from ... import models
from ...data import ExprNotation, TPEFormat
from ...optimization import lr_schedules
from ..ltl_spec import LTLSpecTreeEncoder
from .ltl_syn_experiment import LTLSynExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSynTransformerExperiment(LTLSynExperiment):
    def __init__(
        self,
        constant_learning_rate: float = None,
        custom_pos_enc: bool = True,
        d_embed: int = 256,
        d_embed_enc: int = None,
        d_embed_dec: int = None,
        d_ff: int = 1024,
        dropout: float = 0.0,
        ff_activation: str = "relu",
        name: str = "transformer",
        num_heads: int = 4,
        num_layers: int = 8,
        num_layers_enc: int = None,
        num_layers_dec: int = None,
        warmup_steps: int = 4000,
        **kwargs,
    ):
        self._attn_model = None
        self.constant_learning_rate = constant_learning_rate
        self.custom_pos_enc = custom_pos_enc
        if not custom_pos_enc:
            raise NotImplementedError
        self.d_embed_enc = d_embed_enc if d_embed_enc else d_embed
        self.d_embed_dec = d_embed_dec if d_embed_dec else d_embed
        self.d_ff = d_ff
        self.dropout = dropout
        self.ff_activation = ff_activation
        self.num_heads = num_heads
        self.num_layers_enc = num_layers_enc if num_layers_enc else num_layers
        self.num_layers_dec = num_layers_dec if num_layers_dec else num_layers
        self.warmup_steps = warmup_steps
        if self.d_embed_enc % self.num_heads != 0:
            sys.exit(
                f"Encoder embedding dimension {self.d_embed_enc} is "
                "not divisible by the number of attention heads"
                f"{self.num_heads}"
            )
        if self.d_embed_dec % self.num_heads != 0:
            sys.exit(
                (
                    f"Decoder embedding dimension {self.d_embed_dec} is "
                    "not divisible by the number of attention heads "
                    f"{self.num_heads}"
                )
            )
        super().__init__(name=name, **kwargs)

    @property
    def attn_model(self):
        if not self._attn_model:
            self._attn_model = self.init_model(training=False, attn_weights=True)
            logger.info("Created attention model")
            checkpoint = tf.train.latest_checkpoint(self.local_path(self.name))
            if checkpoint:
                logger.info("Found checkpoint %s", checkpoint)
                self._attn_model.load_weights(checkpoint).expect_partial()
                logger.info("Loaded weights from checkpoint")
        return self._attn_model

    def attn_weights(self, spec, training: bool = False):
        if not self.input_encoder.encode(spec):
            logger.info("Econding error: %s", self.input_encoder.error)
            return None
        formula_tensor, pos_enc_tensor = self.input_encoder.tensor
        # pylint: disable=E1102
        preds, _, enc_attn, dec_attn = self.attn_model(
            (tf.expand_dims(formula_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
            training=training,
        )
        results = []
        num_tokens = len(self.input_encoder.tokens)
        attention_dict_list = []
        for head in range(0, self.num_heads):
            # f = open(f"att{head}.json","a")
            # f.write(str(self.input_encoder.tokens)+"\n")
            layerdict = {}
            for layer in range(1, self.num_layers_enc + 1):
                playerdict = {}
                for player in range(0, num_tokens):
                    attended_player_dict = {}
                    for player_attended in range(0, num_tokens):
                        att = enc_attn[f"layer_{layer}"]["self_attn"][0][head][player][
                            player_attended
                        ].numpy()
                        attended_player_dict[player_attended] = str(att)
                        # f.write(f"Layer {layer}:\n")
                        # f.write(f"Player {player} attends to Player {player_attended}:\n")
                        # f.write(str(att)+"\n")
                        # print(att)
                    playerdict[player] = attended_player_dict
                layerdict[layer] = playerdict
            attention_dict_list.append(layerdict)
            # json.dump(layerdict,f)
            # f.close()

        for beam in preds[0]:
            if not self.target_encoder.decode(np.array(beam)):
                logger.info("Decoding error: %s", self.target_encoder.error)
                # return None
            beam_result = {}
            beam_result["circuit"] = self.target_encoder.circuit
            results.append(beam_result)
        return (attention_dict_list, self.num_layers_enc, self.input_encoder.tokens)

    @property
    def init_input_encoder(self):
        return LTLSpecTreeEncoder(
            notation=ExprNotation.INFIX,
            encoded_notation=ExprNotation.PREFIX,
            start=False,
            eos=False,
            pad=self.max_input_length,
            tpe_format=TPEFormat.BRANCHDOWN,
            tpe_pad=self.d_embed_enc,
        )

    @property
    def init_learning_rate(self):
        if self.constant_learning_rate:
            return self.constant_learning_rate
        return lr_schedules.TransformerSchedule(self.d_embed_enc, warmup_steps=self.warmup_steps)

    @property
    def init_optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

    def init_model(self, training: bool = True, attn_weights: bool = False):
        args = self.args
        args["input_vocab_size"] = self.input_vocab_size
        args["input_eos_id"] = self.input_eos_id
        args["input_pad_id"] = self.input_pad_id
        args["target_vocab_size"] = self.target_vocab_size
        args["target_start_id"] = self.target_start_id
        args["target_eos_id"] = self.target_eos_id
        args["target_pad_id"] = self.target_pad_id
        args["max_encode_length"] = self.max_input_length
        args["max_decode_length"] = self.max_target_length
        return models.transformer.create_model(
            args, training=training, custom_pos_enc=self.custom_pos_enc, attn_weights=attn_weights
        )

    def prepare_tf_dataset(self, tf_dataset):
        def shape_dataset(input_tensor, target_tensor):
            if self.custom_pos_enc:
                ltl_tensor, tpe_tensor = input_tensor
                return ((ltl_tensor, tpe_tensor, target_tensor), target_tensor)
            return ((input_tensor, target_tensor), target_tensor)

        return tf_dataset.map(shape_dataset)

    @classmethod
    def add_init_args(cls, parser):
        super().add_init_args(parser)
        defaults = cls.get_default_args()
        parser.add_argument(
            "--constant-learning-rate", type=float, default=defaults["constant_learning_rate"]
        )
        parser.add_argument("--d-embed", type=int, default=defaults["d_embed"])
        parser.add_argument("--d-embed-enc", type=int, default=defaults["d_embed_enc"])
        parser.add_argument("--d-embed-dec", type=int, default=defaults["d_embed_dec"])
        parser.add_argument("--d-ff", type=int, default=defaults["d_ff"])
        parser.add_argument("--dropout", type=float, default=defaults["dropout"])
        parser.add_argument("--ff-activation", type=str, default=defaults["ff_activation"])
        parser.add_argument("--num-heads", type=int, default=defaults["num_heads"])
        parser.add_argument("--num-layers", type=int, default=defaults["num_layers"])
        parser.add_argument("--num-layers-enc", type=int, default=defaults["num_layers_enc"])
        parser.add_argument("--num-layers-dec", type=int, default=defaults["num_layers_dec"])
        parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])


if __name__ == "__main__":
    LTLSynTransformerExperiment.cli()
