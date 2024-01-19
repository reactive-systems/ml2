"""CSV dataset from evaluation data"""

import logging
from copy import deepcopy
from functools import cmp_to_key
from typing import List, Type, Union

import pandas as pd

from ...aiger import AIGERCircuit
from ...datasets import CSVDataset
from ...dtypes import CSVDict
from ..ltl_spec import DecompLTLSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPEC_STATS = ["size", "num_properties", "max_prop_length", "num_aps"]
CIRC_STATS = ["num_latches", "num_ands", "max_var_id"]


class LTLSynEvalDataset(CSVDataset[CSVDict]):
    def __init__(
        self,
        dtype: Type[CSVDict],
        df: pd.DataFrame = None,
        add_stats: bool = True,
        **kwargs,
    ):
        df.reset_index(drop=True, inplace=True)
        super().__init__(dtype=dtype, df=df, **kwargs)
        if add_stats:
            self.df["verification_satisfied"] = self.df["verification_satisfied"].apply(
                lambda x: x == 1 or x == "1"
            )
            self.df["prediction_valid"] = self.df["verification_satisfied"]
            self.add_circ_stats()
            self.add_spec_stats()
            self.add_par_time()

    def add_par_time(self):
        beam_size = int(len(self.df) / len(self.df.groupby("input_id_DecompLTLSpec")))
        self.df["syn_time_par"] = self.df.apply(
            lambda row: (
                float(row["syn_time"]) / beam_size
                + (
                    float(row["ver_time"])
                    if (not pd.isnull(row["ver_time"]) and row["ver_time"] != "")
                    else 0
                )
                if row["syn_time"] != ""
                else ""
            ),
            axis=1,
        )

    @staticmethod
    def _add_spec_stats(df: pd.DataFrame, prefix: str = "", suffix: str = ""):
        cols = [prefix + e + suffix for e in SPEC_STATS]

        def add_spec_stats_row(row, prefix: str, suffix: str):
            spec: DecompLTLSpec = DecompLTLSpec.from_csv_fields(
                row.to_dict(), prefix=prefix, suffix=suffix
            )
            try:
                max_prop_length = 0
                size = len(spec.guarantees) - 1 + max(len(spec.assumptions) - 1, 0)
                [size := size + g.size() for g in spec.guarantees]
                [max_prop_length := max(max_prop_length, g.size()) for g in spec.guarantees]
                if len(spec.assumptions) != 0:
                    [size := size + a.size() for a in spec.assumptions]
                    [max_prop_length := max(max_prop_length, a.size()) for a in spec.assumptions]
                    size += 1
            except RecursionError:
                size = "inf"
                max_prop_length = "inf"

            return {
                "spec_size": size,
                "num_properties": len(spec.guarantees) + len(spec.assumptions),
                "max_prop_length": max_prop_length,
                "num_aps": len(spec.inputs) + len(spec.outputs),
            }

        df[cols] = df.apply(
            lambda row: add_spec_stats_row(row, prefix=prefix, suffix=suffix),
            axis=1,
            result_type="expand",
        )
        return df

    def add_spec_stats(self):
        if "input_assumptions" in self.df.columns and "input_guarantees" in self.df.columns:
            self.df = self._add_spec_stats(self.df, prefix="input_")

    @staticmethod
    def _add_circ_stats(df: pd.DataFrame, prefix: str = "", suffix: str = ""):
        cols = [prefix + e + suffix for e in CIRC_STATS]

        def add_circ_stats_row(row, prefix: str, suffix: str):
            try:
                circ: AIGERCircuit = AIGERCircuit.from_csv_fields(
                    row.to_dict(), prefix=prefix, suffix=suffix
                )
            except Exception:
                circ = None
            if circ is not None:
                return {
                    "num_latches": circ.num_latches,
                    "num_ands": circ.num_ands,
                    "max_var_id": circ.max_var_id,
                }
            else:
                return {}

        df[cols] = df.apply(
            lambda row: add_circ_stats_row(row, prefix=prefix, suffix=suffix),
            axis=1,
            result_type="expand",
        )
        return df

    def add_circ_stats(self):
        if "prediction_circuit" in self.df.columns:
            self.df = self._add_circ_stats(self.df, prefix="prediction_")
        if "target_circuit" in self.df.columns:
            self.df = self._add_circ_stats(self.df, prefix="target_")

    def _select_rows(
        self,
        smallest: bool,
        fastest: bool,
        inplace: bool,
        par: bool = True,
    ):
        def compare(row1, row2):
            dif = row1["prediction_num_latches"] - row2["prediction_num_latches"]
            if dif != 0:
                return dif
            else:
                return row1["prediction_num_ands"] - row2["prediction_num_ands"]

        def select_row(df, fastest: bool, smallest: bool) -> int:
            if (fastest and smallest) or (not smallest and not fastest):
                raise ValueError
            df_v = df[df["prediction_valid"]]
            if len(df_v) > 1:
                if fastest:
                    if par:
                        return df_v["syn_time_par"].idxmin()
                    else:
                        return df_v["syn_time"].idxmin()
                elif smallest:
                    s_list = sorted((row for _, row in df_v.iterrows()), key=cmp_to_key(compare))
                    return s_list[0].name
            elif len(df_v) == 1:
                return df_v.index.to_list()[0]
            else:
                return df.first_valid_index()

        if inplace:
            obj = self
        else:
            obj = deepcopy(self)

        new_df = obj.df.loc[
            [
                select_row(row, fastest=fastest, smallest=smallest)
                for row in (
                    (obj.df.loc[y])
                    for _, y in obj.df.groupby("input_id_DecompLTLSpec").groups.items()
                )
            ]
        ]

        obj.df = new_df

        return obj

    def group_agg_fastest(self, inplace: bool, par: bool = True):
        return self._select_rows(smallest=False, fastest=True, inplace=inplace, par=par)

    def group_agg_smallest(self, inplace: bool):
        return self._select_rows(smallest=True, fastest=False, inplace=inplace)

    def get(
        self,
        reference: Union[str, pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        """Gets the DataFrame containing results for all solvers (or a given solver) with the given unique references.

        Args:
            reference (Union[str, pd.Series, pd.DataFrame]): Either the unique name of the sample (column input_name) or the row of the sample (i.e. Series) or a DataFrame of multiple samples.

        Returns:
            pd.DataFrame: Dataframe consisting of the requested samples
        """
        if isinstance(reference, str):
            df = self.df[self.df["input_id_DecompLTLSpec"] == reference]
        elif isinstance(reference, pd.Series):
            df = self.df[
                self.df["input_id_DecompLTLSpec"].isin([reference["input_id_DecompLTLSpec"]])
            ]
        elif isinstance(reference, pd.DataFrame):
            df = self.df[
                self.df["input_id_DecompLTLSpec"].isin(list(reference["input_id_DecompLTLSpec"]))
            ]
        return df

    @classmethod
    def from_merge(cls, evaluations: List["LTLSynEvalDataset"], **kwargs) -> "LTLSynEvalDataset":
        df = pd.concat([eval.df for eval in evaluations])
        metadata = {
            "joined_from": [{**eval.metadata, "name": eval.bucket_path} for eval in evaluations]
        }

        return cls(dtype=CSVDict, df=df, metadata=metadata, add_stats=False, **kwargs)
