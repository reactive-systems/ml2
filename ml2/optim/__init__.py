from ..utils import is_tf_available

if is_tf_available():
    from . import tf_optim
