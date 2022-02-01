"""LTL specification encoder"""

import tensorflow as tf

from ...data.encoder import PAD_TOKEN
from ..ltl_encoder import LTLTreeEncoder
from .ltl_spec import LTLSpec


class LTLSpecTreeEncoder(LTLTreeEncoder):
    def encode(self, spec: LTLSpec) -> bool:
        return super().encode(spec.formula_str)


class LTLSpecGuaranteeEncoder(LTLTreeEncoder):
    def __init__(self, guarantee_pad: int, num_guarantees: int, **kwargs):
        self.num_guarantees = num_guarantees
        self.guarantee_ids = []
        self.guarantee_padded_tpe = []
        super().__init__(pad=guarantee_pad, **kwargs)

    @property
    def tensor_spec(self):
        guarantee_spec = tf.TensorSpec(shape=(self.num_guarantees, self.pad), dtype=self.tf_dtype)
        tpe_spec = tf.TensorSpec(
            shape=(self.num_guarantees, self.pad, self.tpe_pad), dtype=self.tf_dtype
        )
        return (guarantee_spec, tpe_spec)

    @property
    def tensor(self):
        guarantee_tensor = tf.constant(self.guarantee_ids, dtype=self.tf_dtype)
        tpe_tensor = tf.constant(self.guarantee_padded_tpe, dtype=self.tf_dtype)
        return (guarantee_tensor, tpe_tensor)

    def encode(self, spec: LTLSpec) -> bool:
        if len(spec.guarantees) > self.num_guarantees:
            self.error = "Num guarantees"
            return False
        self.guarantee_ids = []
        self.guarantee_padded_tpe = []
        for guarantee in spec.guarantees:
            if not super().encode(guarantee):
                return False
            self.guarantee_ids.append(self.ids)
            self.guarantee_padded_tpe.append(self.padded_tpe)
        for _ in range(len(spec.guarantees), self.num_guarantees):
            if self.vocabulary:
                self.guarantee_ids.append(self.vocabulary.tokens_to_ids([PAD_TOKEN] * self.pad))
                self.guarantee_padded_tpe.append([[0] * self.tpe_pad] * self.pad)
        return True


class LTLSpecPropertyEncoder(LTLTreeEncoder):
    def __init__(self, property_pad: int, num_properties: int, **kwargs):
        self.num_properties = num_properties
        self.property_ids = []
        self.property_padded_tpe = []
        self.property_tokens = []
        self.property_padded_tokens = []
        super().__init__(start=True, pad=property_pad, **kwargs)

    @property
    def tensor_spec(self):
        property_spec = tf.TensorSpec(shape=(self.num_properties, self.pad), dtype=self.tf_dtype)
        tpe_spec = tf.TensorSpec(
            shape=(self.num_properties, self.pad, self.tpe_pad), dtype=self.tf_dtype
        )
        return (property_spec, tpe_spec)

    @property
    def tensor(self):
        property_tensor = tf.constant(self.property_ids, dtype=self.tf_dtype)
        tpe_tensor = tf.constant(self.property_padded_tpe, dtype=self.tf_dtype)
        return (property_tensor, tpe_tensor)

    def encode(self, spec: LTLSpec) -> bool:
        if len(spec.assumptions + spec.guarantees) > self.num_properties:
            self.error = "Num properties"
            return False
        self.property_ids = []
        self.property_padded_tpe = []
        self.property_tokens = []
        self.property_padded_tokens = []
        self.encode_start = True
        for assumption in spec.assumptions:
            if not super().encode(assumption):
                return False
            self.property_ids.append(self.ids)
            self.property_padded_tpe.append(self.padded_tpe)
            self.property_tokens.append(self.tokens)
            self.property_padded_tokens.append(self.padded_tokens)
        self.encode_start = False
        for guarantee in spec.guarantees:
            if not super().encode(guarantee):
                return False
            self.property_ids.append(self.ids)
            self.property_padded_tpe.append(self.padded_tpe)
            self.property_tokens.append(self.tokens)
            self.property_padded_tokens.append(self.padded_tokens)
        for _ in range(len(spec.assumptions + spec.guarantees), self.num_properties):
            if self.vocabulary:
                self.property_ids.append(self.vocabulary.tokens_to_ids([PAD_TOKEN] * self.pad))
                self.property_padded_tpe.append([[0] * self.tpe_pad] * self.pad)
        return True
