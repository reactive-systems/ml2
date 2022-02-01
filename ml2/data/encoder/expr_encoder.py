"""Expression encoder"""

import copy
import tensorflow as tf

from .seq_encoder import SeqEncoder
from ..ast import TPEFormat
from ..expr import ExprNotation
from ..vocabulary import Vocabulary


class ExprEncoder(SeqEncoder):
    def __init__(
        self,
        notation: ExprNotation,
        encoded_notation: ExprNotation,
        start: bool,
        eos: bool,
        pad: int,
        encode_start: bool = True,
        tpe_format: TPEFormat = None,
        tpe_pad: int = None,
        vocabulary: Vocabulary = None,
        tf_dtype: tf.DType = tf.int32,
    ):
        """
        Args:
            notation: ml2.data.expr.ExprNotation, notation of expressions that are encoded
            encoded_notation: ml2.data.expr.ExprNotation, notation of encoding
            tpe_format: ml2.data.ast.TPEFormat or None, format of tree positional encoding
            tpe_pad: tree positional encoding padding corresponding to the embedding dimension or None for no padding
            see ml2.data.encoder.SeqEncoder for other arguments
        """
        self.notation = notation
        self.encoded_notation = encoded_notation
        self.tpe_format = tpe_format
        self.tpe_pad = tpe_pad
        self.tf_dtype = tf_dtype

        self.ast = None
        self.tpe = None
        self.padded_tpe = None
        super().__init__(start, eos, pad, encode_start, vocabulary, tf_dtype)

    def lex(self) -> bool:
        raise NotImplementedError

    def parse(self) -> bool:
        raise NotImplementedError

    @property
    def tensor_spec(self):
        expression_spec = tf.TensorSpec(shape=(self.pad,), dtype=self.tf_dtype)
        if self.tpe_format:
            tpe_spec = tf.TensorSpec(shape=(self.pad, self.tpe_pad), dtype=self.tf_dtype)
            return (expression_spec, tpe_spec)
        else:
            return expression_spec

    @property
    def tensor(self):
        expression_tensor = tf.constant(self.ids, dtype=self.tf_dtype)
        if self.tpe_format:
            tpe_tensor = tf.constant(self.padded_tpe, dtype=self.tf_dtype)
            return (expression_tensor, tpe_tensor)
        else:
            return expression_tensor

    @property
    def expression(self):
        return self.sequence

    def tokenize(self) -> bool:
        success = self.parse()
        if success:
            self.tokens = self.ast.to_list(self.encoded_notation)
        success = success and self.compute_tpe()
        return success

    def detokenize(self) -> bool:
        # TODO add functionality to decode from tokens and tree positional encoding
        super().detokenize()
        success = self.parse()
        if success:
            self.sequence = self.ast.to_str(self.notation)
        return success

    def compute_tpe(self) -> bool:
        self.tpe = self.ast.tree_positional_encoding(self.encoded_notation, self.tpe_format)
        self.padded_tpe = copy.deepcopy(self.tpe)
        if self.start and self.encode_start:
            self.padded_tpe.insert(0, [])
        if self.eos:
            self.padded_tpe.append([])
        if self.pad:
            if self.pad < len(self.padded_tpe):
                self.error = "Token TPE padding"
                return False
            self.padded_tpe.extend([[]] * (self.pad - len(self.padded_tpe)))
        if self.tpe_pad:
            for l in self.padded_tpe:
                if self.tpe_pad < len(l):
                    self.error = "TPE padding"
                    return False
                l.extend([0] * (self.tpe_pad - len(l)))
        return True

    def vocabulary_filename(self) -> str:
        return "-" + self.encoded_notation.value + super().vocabulary_filename()
