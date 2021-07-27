"""LTL synthesis using hierarchical Transformer"""

import logging
import sys
import tensorflow as tf

from ... import models

from ...data import TPEFormat
from ...data import ExprNotation
from ...optimization import lr_schedules
from ..ltl_spec import LTLSpecPropertyEncoder
from .ltl_syn_experiment import LTLSynExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSynHierTransformerExperiment(LTLSynExperiment):

    def __init__(self,
                 constant_learning_rate: float = None,
                 custom_pos_enc: bool = True,
                 d_embed_enc: int = 256,
                 d_embed_dec: int = None,
                 d_ff_enc_d0: int = 1024,
                 d_ff_enc_d1: int = 1024,
                 d_ff_dec: int = 1024,
                 dropout_enc: float = 0.0,
                 dropout_dec: float = 0.0,
                 ff_activation_enc_d0: str = 'relu',
                 ff_activation_enc_d1: str = 'relu',
                 ff_activation_dec: str = 'relu',
                 name: str = 'hier-transformer',
                 num_properties: int = 12,
                 num_heads_enc_d0: int = 4,
                 num_heads_enc_d1: int = 4,
                 num_heads_dec: int = 4,
                 num_layers_enc_d0: int = 4,
                 num_layers_enc_d1: int = 4,
                 num_layers_dec: int = None,
                 property_tree_size: int = 25,
                 warmup_steps: int = 4000,
                 **kwargs):
        self.constant_learning_rate = constant_learning_rate
        self.custom_pos_enc = custom_pos_enc
        if not custom_pos_enc:
            raise NotImplementedError
        self.d_embed_enc = d_embed_enc
        self.d_embed_dec = d_embed_dec if d_embed_dec else self.d_embed_enc
        self.d_ff_enc_d0 = d_ff_enc_d0
        self.d_ff_enc_d1 = d_ff_enc_d1
        self.d_ff_dec = d_ff_dec
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.ff_activation_enc_d0 = ff_activation_enc_d0
        self.ff_activation_enc_d1 = ff_activation_enc_d1
        self.ff_activation_dec = ff_activation_dec
        self.property_tree_size = property_tree_size
        self.num_properties = num_properties
        self.num_heads_enc_d0 = num_heads_enc_d0
        self.num_heads_enc_d1 = num_heads_enc_d1
        self.num_heads_dec = num_heads_dec
        self.num_layers_enc_d0 = num_layers_enc_d0
        self.num_layers_enc_d1 = num_layers_enc_d1
        self.num_layers_dec = num_layers_dec if num_layers_dec else self.num_layers_enc_d0 + self.num_layers_enc_d1
        self.warmup_steps = warmup_steps
        if self.d_embed_enc % self.num_heads_enc_d0 != 0:
            sys.exit(f'Encoder embedding dimension {self.d_embed_enc} is '
                     'not divisible by the number of attention heads'
                     f'{self.num_heads_enc_d0}')
        if self.d_embed_enc % self.num_heads_enc_d1 != 0:
            sys.exit(f'Encoder embedding dimension {self.d_embed_enc} is '
                     'not divisible by the number of attention heads'
                     f'{self.num_heads_enc_d1}')
        if self.d_embed_dec % self.num_heads_dec != 0:
            sys.exit((f'Decoder embedding dimension {self.d_embed_dec} is '
                      'not divisible by the number of attention heads '
                      f'{self.num_heads_dec}'))
        super().__init__(name=name, **kwargs)

    @property
    def init_input_encoder(self):
        return LTLSpecPropertyEncoder(property_pad=self.property_tree_size,
                                      num_properties=self.num_properties,
                                      notation=ExprNotation.INFIX,
                                      encoded_notation=ExprNotation.PREFIX,
                                      eos=False,
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

    def init_model(self, training=True):
        args = self.args
        args['input_vocab_size'] = self.input_vocab_size
        args['input_eos_id'] = self.input_eos_id
        args['input_pad_id'] = self.input_pad_id
        args['target_vocab_size'] = self.target_vocab_size
        args['target_start_id'] = self.target_start_id
        args['target_eos_id'] = self.target_eos_id
        args['target_pad_id'] = self.target_pad_id
        args['input_length'] = (self.num_properties, self.property_tree_size)
        args['max_decode_length'] = self.max_target_length
        return models.hierarchical_transformer_2d.create_model(
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
        parser.add_argument('--d-ff-enc-d0',
                            type=int,
                            default=defaults['d_ff_enc_d0'])
        parser.add_argument('--d-ff-enc-d1',
                            type=int,
                            default=defaults['d_ff_enc_d1'])
        parser.add_argument('--d-ff-dec',
                            type=int,
                            default=defaults['d_ff_dec'])
        parser.add_argument('--dropout-enc',
                            type=float,
                            default=defaults['dropout_enc'])
        parser.add_argument('--dropout-dec',
                            type=float,
                            default=defaults['dropout_dec'])
        parser.add_argument('--ff-activation-enc-d0',
                            type=str,
                            default=defaults['ff_activation_enc_d0'])
        parser.add_argument('--ff-activation-enc-d1',
                            type=str,
                            default=defaults['ff_activation_enc_d1'])
        parser.add_argument('--ff-activation-dec',
                            type=str,
                            default=defaults['ff_activation_dec'])
        parser.add_argument('--num-heads-enc-d0',
                            type=int,
                            default=defaults['num_heads_enc_d0'])
        parser.add_argument('--num-heads-enc-d1',
                            type=int,
                            default=defaults['num_heads_enc_d1'])
        parser.add_argument('--num-heads-dec',
                            type=int,
                            default=defaults['num_heads_dec'])
        parser.add_argument('--num-layers-enc-d0',
                            type=int,
                            default=defaults['num_layers_enc_d0'])
        parser.add_argument('--num-layers-enc-d1',
                            type=int,
                            default=defaults['num_layers_enc_d1'])
        parser.add_argument('--num-layers-dec',
                            type=int,
                            default=defaults['num_layers_dec'])
        parser.add_argument('--warmup-steps',
                            type=int,
                            default=defaults['warmup_steps'])


if __name__ == '__main__':
    LTLSynHierTransformerExperiment.cli()
