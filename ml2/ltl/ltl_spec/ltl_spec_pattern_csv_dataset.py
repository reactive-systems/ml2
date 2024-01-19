"""LTL specification pattern CSV dataset"""

import argparse
import logging
from copy import deepcopy
from typing import Optional, Tuple

from ...datasets.split_dataset import SplitDataset
from ...globals import LTL_SPEC_ALIASES, LTL_SPEC_PROJECT_NAME
from ...registry import register_type
from .ltl_spec import LTLSpec
from .ltl_spec_csv_dataset import LTLSpecCSVDataset
from .ltl_spec_dataset import LTLSpecDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class LTLSpecPatternCSVDataset(SplitDataset[LTLSpec, LTLSpecCSVDataset]):
    ALIASES = LTL_SPEC_ALIASES

    def filter(
        self,
        ast_size: Tuple[Optional[int], Optional[int]] = (None, None),
        num_inputs: Tuple[Optional[int], Optional[int]] = (None, None),
        num_outputs: Tuple[Optional[int], Optional[int]] = (None, None),
        inplace: bool = False,
    ) -> Optional["LTLSpecPatternCSVDataset"]:
        if not inplace:
            result = deepcopy(self)
        else:
            result = {}
        for split in self.split_names:
            result[split] = self[split].filter(  # type: ignore
                ast_size=ast_size,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                inplace=inplace,
            )
        if not inplace:
            return result  # type: ignore

    @classmethod
    def from_ltl_spec_data(
        cls, data: LTLSpecDataset, name: str, project: Optional[str] = None, **kwargs
    ) -> "LTLSpecPatternCSVDataset":
        if project is None:
            project = LTL_SPEC_PROJECT_NAME
        return cls(name=name, splits=data.properties(), dtype=LTLSpec, project=project, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts assumption and guarantee patterns from LTL specification data and writes them to a file"
    )
    parser.add_argument("--ltl-spec-data", help="LTL specification data")
    parser.add_argument("--name", help="name of the extracted LTL specification pattern dataset")
    args = parser.parse_args()

    ltl_spec_data: LTLSpecDataset = LTLSpecDataset.load(args.ltl_spec_data)  # type: ignore
    ltl_spec_pattern_data: LTLSpecPatternCSVDataset = LTLSpecPatternCSVDataset.from_ltl_spec_data(
        data=ltl_spec_data, name=args.name
    )
    ltl_spec_pattern_data.save(name=args.name, auto_version=True, upload=True)
