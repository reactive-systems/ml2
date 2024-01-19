from ..utils import is_pt_available, is_tf_available
from .mealy_machine import Condition, HoaHeader, MealyMachine, Transition

if is_pt_available() and is_tf_available():
    from .mealy_tokenizer import MealyToSeqTokenizer
