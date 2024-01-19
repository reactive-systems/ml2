"""sc-0 dataset test"""

from ml2.datasets import load_dataset
from ml2.ltl.ltl_spec import DecompLTLSpec


def test_load_sc_0():
    ds = load_dataset("ltl-spec/sc-0")
    assert ds.name == "sc-0"
    assert ds.project == "ltl-spec"
    assert ds.dtype == DecompLTLSpec
    assert ds.size == 346
