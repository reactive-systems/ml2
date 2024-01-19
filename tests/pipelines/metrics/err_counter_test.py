"""Error counting test"""


from ml2.pipelines.metrics import ErrCounter, EvalSupervisedErrCounter, SupervisedErrCounter
from ml2.pipelines.samples import EncodedSample, EvalLabeledSample, LabeledSample


def test_err_counter():
    metric = ErrCounter()
    sample1 = EncodedSample(inp=None, inp_enc_err=None)
    assert metric.add(sample1) is False

    sample2 = EncodedSample(inp=None, inp_enc_err="Parse error")
    assert metric.add(sample2) is True
    assert metric.compute_dict()["inp_enc_errs"] == 1

    metric.reset()
    assert metric.compute_dict()["inp_enc_errs"] == 0


def test_supervised_err_counter():
    metric = SupervisedErrCounter()
    sample1 = LabeledSample(inp=None, tar=None)
    assert metric.add(sample1) is False

    sample2 = LabeledSample(inp=None, tar=None, inp_enc_err="Parse error")
    assert metric.add(sample2) is True

    sample3 = LabeledSample(inp=None, tar=None, tar_enc_err="Lex error")
    assert metric.add(sample3) is True

    assert metric.compute_dict()["inp_enc_errs"] == 1
    assert metric.compute_dict()["tar_enc_errs"] == 1


def test_eval_supervised_err_counter():
    metric = EvalSupervisedErrCounter()

    sample1 = EvalLabeledSample(inp=None, tar=None, pred_dec_err="Parse error")
    assert metric.add(sample1) is True

    assert metric.compute_dict()["pred_dec_errs"] == 1
