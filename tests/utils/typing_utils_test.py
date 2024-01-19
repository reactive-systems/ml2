import typing

from ml2.datasets import Dataset, SplitDataset
from ml2.ltl import DecompLTLSpec, LTLSpec
from ml2.utils.typing_utils import is_subclass_generic


def test_is_subclass_generic():
    # basic data types
    assert is_subclass_generic(bool, bool)
    assert is_subclass_generic(int, int)
    assert is_subclass_generic(bool, int)
    assert not is_subclass_generic(int, bool)
    # data structures
    assert is_subclass_generic(dict, dict)
    assert is_subclass_generic(tuple, tuple)
    assert not is_subclass_generic(dict, list)
    assert not is_subclass_generic(list, dict)
    # generic data structures
    assert is_subclass_generic(typing.Dict, dict)
    assert is_subclass_generic(tuple, typing.Tuple)
    assert is_subclass_generic(typing.Optional[int], typing.Optional)
    assert is_subclass_generic(typing.Dict[str, int], dict)
    assert is_subclass_generic(typing.Tuple[int, str, int], tuple)
    assert is_subclass_generic(typing.Union[int], int)
    assert is_subclass_generic(typing.Union[int, typing.Union[str]], typing.Union[int, str])
    assert not is_subclass_generic(list, typing.List[int])
    assert not is_subclass_generic(typing.Tuple[int, str], typing.Tuple[int, str, int])
    assert not is_subclass_generic(typing.Tuple[int, str, int], typing.Tuple[int, str])
    # generic data structures with ml2 data types
    assert is_subclass_generic(typing.List[DecompLTLSpec], typing.List[LTLSpec])
    assert is_subclass_generic(typing.Union[DecompLTLSpec], LTLSpec)
    assert is_subclass_generic(
        typing.Tuple[DecompLTLSpec, DecompLTLSpec], typing.Tuple[LTLSpec, LTLSpec]
    )
    assert not is_subclass_generic(typing.Optional[LTLSpec], typing.Optional[DecompLTLSpec])
    assert not is_subclass_generic(
        typing.Mapping[int, LTLSpec], typing.Mapping[int, DecompLTLSpec]
    )
    # nested generic data structures
    assert is_subclass_generic(typing.List[typing.List[int]], list)
    assert not is_subclass_generic(list, typing.List[typing.List[int]])
    assert not is_subclass_generic(typing.List[typing.List], typing.List[typing.List[int]])
    assert is_subclass_generic(
        typing.List[typing.Tuple[DecompLTLSpec, DecompLTLSpec]],
        typing.List[typing.Tuple[LTLSpec, LTLSpec]],
    )
    assert not is_subclass_generic(
        typing.List[typing.Tuple[LTLSpec, LTLSpec]],
        typing.List[typing.Tuple[LTLSpec, DecompLTLSpec]],
    )
    # generic data structures from ml2
    assert is_subclass_generic(Dataset[LTLSpec], Dataset)
    assert is_subclass_generic(Dataset[DecompLTLSpec], Dataset[LTLSpec])
    assert not is_subclass_generic(Dataset, Dataset[LTLSpec])
    assert not is_subclass_generic(Dataset[LTLSpec], Dataset[DecompLTLSpec])
    # type variables
    UNBOUND_VAR = typing.TypeVar("UNBOUND_VAR")
    SPEC_VAR = typing.TypeVar("SPEC_VAR", bound=LTLSpec)
    DECOMP_SPEC_VAR = typing.TypeVar("DECOMP_SPEC_VAR", bound=DecompLTLSpec)
    assert is_subclass_generic(SPEC_VAR, UNBOUND_VAR)
    assert is_subclass_generic(DECOMP_SPEC_VAR, SPEC_VAR)
    assert not is_subclass_generic(SPEC_VAR, DECOMP_SPEC_VAR)
    # nested generic data structures and type variables
    SpecDataset = Dataset[SPEC_VAR]
    DecompSpecDataset = Dataset[DECOMP_SPEC_VAR]
    SPEC_DATASET_VAR = typing.TypeVar("SPEC_DATASET_VAR", bound=SpecDataset)
    DECOMP_SPEC_DATASET_VAR = typing.TypeVar("DECOMP_SPEC_DATASET_VAR", bound=DecompSpecDataset)
    assert is_subclass_generic(
        SplitDataset[DECOMP_SPEC_VAR, DECOMP_SPEC_DATASET_VAR],
        SplitDataset[SPEC_VAR, SPEC_DATASET_VAR],
    )
    assert not is_subclass_generic(
        SplitDataset[SPEC_VAR, SPEC_DATASET_VAR],
        SplitDataset[DECOMP_SPEC_VAR, DECOMP_SPEC_DATASET_VAR],
    )
