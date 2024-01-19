"""Accuracy test"""

from ml2.pipelines.metrics import Acc
from ml2.pipelines.samples import EvalLabeledSample


def test_acc():
    metric = Acc()
    sample1 = EvalLabeledSample(inp=None, tar=None, tar_enc=[0, 0, 1], pred_enc=[0, 0, 0])
    assert metric.add(sample1) == 2 / 3

    sample2 = EvalLabeledSample(inp=None, tar=None, tar_enc=[2, 2, 2], pred_enc=[2, 2, 2])
    assert metric.add(sample2) == 1.0
    assert metric.compute() == 5 / 6

    metric.reset()
    assert metric.compute() == 0.0

    sample3 = EvalLabeledSample(inp=None, tar=None, tar_enc=[-1], pred_enc=[-1])
    assert metric.add(sample3) == 1.0
    assert metric.compute() == 1.0


def test_pad_acc():
    metric = Acc(pad_same_length=True)

    sample1 = EvalLabeledSample(inp=None, tar=None, tar_enc=[1, 1, 0, 0], pred_enc=[1, 1])
    assert metric.add(sample1) == 1.0

    sample2 = EvalLabeledSample(inp=None, tar=None, tar_enc=[1, 3, 2, 1], pred_enc=[1, 1, 1])
    assert metric.add(sample2) == 1 / 4

    assert metric.compute() == 5 / 8

    metric.reset()

    assert metric.compute() == 0.0
