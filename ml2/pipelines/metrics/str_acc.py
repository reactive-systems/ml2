"""String accuracy"""


from ..samples import LabeledSample
from .metric import Metric


class StrAcc(Metric):
    def __init__(self, count_none: bool = True, name: str = "str-acc") -> None:
        self.count_none = count_none
        self.acc_not_norm = 0
        self.count = 0
        super().__init__(name=name)

    def add(self, sample: LabeledSample) -> bool:
        if sample.tar is None or sample.pred is None:
            if self.count_none:
                self.count += 1
            return False

        self.count += 1
        if sample.tar.to_str() == sample.pred.to_str():
            self.acc_not_norm += 1
            return True
        else:
            return False

    def compute(self) -> float:
        if self.count > 0:
            return self.acc_not_norm / self.count
        return 0.0

    def reset(self) -> None:
        self.acc_not_norm = 0
        self.count = 0
