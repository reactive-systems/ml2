"""Accuracy per sequence test"""

from ml2.pipelines.metrics import AccPerSeq
from ml2.pipelines.samples import EvalLabeledSample


def test_acc_add():
    metric = AccPerSeq()
    sample1 = EvalLabeledSample(inp=None, tar=None, tar_enc=[0, 0, 0], pred_enc=[0, 0, 0])
    assert metric.add(sample1) == 1

    sample2 = EvalLabeledSample(inp=None, tar=None, tar_enc=[1, 1, 1], pred_enc=[0, 1, 1])
    assert metric.add(sample2) == 0
    assert metric.compute() == 0.5

    metric.reset()
    assert metric.compute() == 0.0

    sample3 = EvalLabeledSample(inp=None, tar=None, tar_enc=[0], pred_enc=[1])
    assert metric.add(sample3) == 0.0
    assert metric.compute() == 0.0
