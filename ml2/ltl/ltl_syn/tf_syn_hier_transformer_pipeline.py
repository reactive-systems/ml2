"""TensorFlow LTL synthesis hierarchical Transformer pipeline"""

import logging
from typing import Optional, TypeVar

from ...dtypes import DType
from ...pipelines.tf_hier_transformer_pipeline import TFHierTransformerPipeline
from ...registry import register_type
from ..ltl_spec.ltl_spec import LTLSpec

I = TypeVar("I", bound=LTLSpec)
T = TypeVar("T", bound=DType)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class TFSynHierTransformerPipeline(TFHierTransformerPipeline[I, T]):
    def decode(self, prediction_encoding, input: Optional[I] = None) -> T:
        assert input is not None
        return self.target_tokenizer.decode(
            prediction_encoding, inputs=input.inputs, outputs=input.outputs
        )
