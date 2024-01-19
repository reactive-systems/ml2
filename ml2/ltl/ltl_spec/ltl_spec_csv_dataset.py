"""LTL specification CSV dataset"""

import logging
from copy import copy
from typing import Dict, Generator, Optional, Tuple

from tqdm import tqdm

from ...datasets.csv_dataset import CSVDataset
from ...registry import register_type
from .ltl_spec import LTLSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_type
class LTLSpecCSVDataset(CSVDataset[LTLSpec]):
    def filter(
        self,
        ast_size: Tuple[Optional[int], Optional[int]] = (None, None),
        num_inputs: Tuple[Optional[int], Optional[int]] = (None, None),
        num_outputs: Tuple[Optional[int], Optional[int]] = (None, None),
        inplace: bool = False,
    ) -> Optional["LTLSpecCSVDataset"]:
        counter = {
            "max_ast_size": 0,
            "max_inputs": 0,
            "max_outputs": 0,
            "min_ast_size": 0,
            "min_inputs": 0,
            "min_outputs": 0,
        }
        t = tqdm(desc="Filtering", total=len(self.df), postfix=counter)

        def filter_pandas(row):
            spec: LTLSpec = LTLSpec.from_csv_fields(row.to_dict())  # type: ignore
            filtered_out: bool = False
            if not (ast_size[0] is None or spec.ast.size() >= ast_size[0]):
                counter["min_ast_size"] += 1
                filtered_out = True
            if not (ast_size[1] is None or spec.ast.size() <= ast_size[1]):
                counter["max_ast_size"] += 1
                filtered_out = True
            if not (num_inputs[0] is None or spec.num_inputs >= num_inputs[0]):
                counter["min_inputs"] += 1
                filtered_out = True
            if not (num_inputs[1] is None or spec.num_inputs <= num_inputs[1]):
                counter["max_inputs"] += 1
                filtered_out = True
            if not (num_outputs[0] is None or spec.num_outputs >= num_outputs[0]):
                counter["min_outputs"] += 1
                filtered_out = True
            if not (num_outputs[1] is None or spec.num_outputs <= num_outputs[1]):
                counter["max_outputs"] += 1
                filtered_out = True
            t.set_postfix(counter, refresh=False)
            t.update()
            return not filtered_out

        result_df = self.df[self.df.apply(filter_pandas, axis=1)].copy()
        result_df.reset_index(inplace=True)
        result_df.drop("index", axis=1, inplace=True)

        logger.info(
            "From filtered %s, %d AST too large, %d AST too small, %d too many inputs,  %d not enough inputs, %d too many outputs, %d not enough outputs. %d specs remaining.",
            self.name,
            counter["max_ast_size"],
            counter["min_ast_size"],
            counter["max_inputs"],
            counter["min_inputs"],
            counter["max_outputs"],
            counter["min_outputs"],
            len(result_df),
        )

        if not inplace:
            new_properties: "LTLSpecCSVDataset" = copy(self)
            new_properties.df = result_df
            return new_properties
        else:
            self.df = result_df

    def generator(
        self,
        rename: bool = False,
        rename_random: bool = False,
        rename_dict: Optional[Dict[str, str]] = None,
        random_weighted: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Generator[LTLSpec, None, None]:
        for _, row in self.df.iterrows():
            row = row.dropna()
            spec: LTLSpec = LTLSpec.from_csv_fields(row.to_dict())  # type: ignore
            if rename or rename_random:
                spec.rename_aps(
                    random=rename_random, renaming=rename_dict, random_weighted=random_weighted
                )
            yield spec
