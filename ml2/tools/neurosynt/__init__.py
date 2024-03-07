from ...utils import is_pt_available, is_tf_available
from .neurosynt import NeuroSynt

if is_pt_available() and is_tf_available():
    from .pipeline_wrapper import PipelineWrapper
