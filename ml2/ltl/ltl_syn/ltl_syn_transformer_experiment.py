"""LTL synthesis using the Transformer"""

import logging
import sys
import tensorflow as tf

from ... import models
from ...data import ExprNotation, TPEFormat
from ...optimization import lr_schedules
from ..ltl_spec import LTLSpecTreeEncoder
from .ltl_syn_experiment import LTLSynExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSynTransformerExperiment(LTLSynExperiment):

    def __init__(self,
                 constant_learning_rate: float = None,
                 custom_pos_enc: bool = True,
                 d_embed_enc: int = 256,
                 d_embed_dec: int = None,
                 d_ff: int = 1024,
                 dropout: float = 0.0,
                 ff_activation: str = 'relu',
                 name: str = 'transformer',
                 num_heads: int = 4,
                 num_layers_enc: int = 8,
                 num_layers_dec: int = None,
                 warmup_steps: int = 4000,
                 **kwargs):
        self.constant_learning_rate = constant_learning_rate
        self.custom_pos_enc = custom_pos_enc
        if not custom_pos_enc:
            raise NotImplementedError
        self.d_embed_enc = d_embed_enc
        self.d_embed_dec = d_embed_dec if d_embed_dec else self.d_embed_enc
        self.d_ff = d_ff
        self.dropout = dropout
        self.ff_activation = ff_activation
        self.num_heads = num_heads
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec if num_layers_dec else self.num_layers_enc
        self.warmup_steps = warmup_steps
        if self.d_embed_enc % self.num_heads != 0:
            sys.exit(f'Encoder embedding dimension {self.d_embed_enc} is '
                     'not divisible by the number of attention heads'
                     f'{self.num_heads}')
        if self.d_embed_dec % self.num_heads != 0:
            sys.exit((f'Decoder embedding dimension {self.d_embed_dec} is '
                      'not divisible by the number of attention heads '
                      f'{self.num_heads}'))
        super().__init__(name=name, **kwargs)

    @property
    def init_input_encoder(self):
        return LTLSpecTreeEncoder(notation=ExprNotation.INFIX,
                                  encoded_notation=ExprNotation.PREFIX,
                                  start=False,
                                  eos=False,
                                  pad=self.max_input_length,
                                  tpe_format=TPEFormat.BRANCHDOWN,
                                  tpe_pad=self.d_embed_enc)

    @property
    def init_learning_rate(self):
        if self.constant_learning_rate:
            return self.constant_learning_rate
        return lr_schedules.TransformerSchedule(self.d_embed_enc,
                                                warmup_steps=self.warmup_steps)

    @property
    def init_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)

    def init_model(self, training: bool = True):
        args = self.args
        args['input_vocab_size'] = self.input_vocab_size
        args['input_eos_id'] = self.input_eos_id
        args['input_pad_id'] = self.input_pad_id
        args['target_vocab_size'] = self.target_vocab_size
        args['target_start_id'] = self.target_start_id
        args['target_eos_id'] = self.target_eos_id
        args['target_pad_id'] = self.target_pad_id
        args['max_encode_length'] = self.max_input_length
        args['max_decode_length'] = self.max_target_length
        return models.transformer.create_model(
            args, training=training, custom_pos_enc=self.custom_pos_enc)

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
        parser.add_argument('--constant-learning-rate',
                            type=float,
                            default=defaults['constant_learning_rate'])
        parser.add_argument('--d-embed-enc',
                            type=int,
                            default=defaults['d_embed_enc'])
        parser.add_argument('--d-embed-dec',
                            type=int,
                            default=defaults['d_embed_dec'])
        parser.add_argument('--d-ff', type=int, default=defaults['d_ff'])
        parser.add_argument('--dropout',
                            type=float,
                            default=defaults['dropout'])
        parser.add_argument('--ff-activation',
                            type=str,
                            default=defaults['ff_activation'])
        parser.add_argument('--num-heads',
                            type=int,
                            default=defaults['num_heads'])
        parser.add_argument('--num-layers-enc',
                            type=int,
                            default=defaults['num_layers_enc'])
        parser.add_argument('--num-layers-dec',
                            type=int,
                            default=defaults['num_layers_dec'])
        parser.add_argument('--warmup-steps',
                            type=int,
                            default=defaults['warmup_steps'])


if __name__ == '__main__':
    LTLSynTransformerExperiment.cli()
