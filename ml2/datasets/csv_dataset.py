"""CSV dataset"""

import csv
import logging
import os
from typing import Any, Dict, Generator, Type, TypeVar

import pandas as pd

from ..dtypes import CSV
from ..registry import register_type
from .dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=CSV)


@register_type
class CSVDataset(Dataset[T]):
    def __init__(
        self,
        dtype: Type[T],
        df: pd.DataFrame = None,
        filename: str = None,
        sep: str = ",",
        **kwargs,
    ):
        self.df = df if df is not None else pd.DataFrame()
        self._filename = filename
        self.sep = sep
        super().__init__(dtype=dtype, **kwargs)

    def __getitem__(self, key: int) -> T:
        return self.dtype.from_csv_fields(self.df.iloc[key].to_dict())

    def add_sample(self, sample: T, **kwargs) -> None:
        df_dictionary = pd.DataFrame([sample.to_csv_fields(**kwargs)])
        self.df = pd.concat([self.df, df_dictionary], ignore_index=True)

    @property
    def filename(self) -> str:
        return self._filename if self._filename is not None else "data.csv"

    def generator(self, **kwargs) -> Generator[T, None, None]:
        for _, row in self.df.iterrows():
            row = row.dropna()
            try:
                yield self.dtype.from_csv_fields(row.to_dict())
            except Exception as err:
                logger.warning(
                    f"Exception {err} on construction of data type from csv fields {row.to_dict()}"
                )

    def config_postprocessors(self) -> list:
        def postprocess_df(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("df", None)
            annotations.pop("df", None)

        return [postprocess_df] + super().config_postprocessors()

    def shuffle(self, seed: int = None) -> None:
        self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)

    @property
    def size(self) -> int:
        return len(self.df.index)

    def save_to_path(self, path: str) -> None:
        self.df.to_csv(
            os.path.join(path, self.filename), index=False, quoting=csv.QUOTE_ALL, sep=self.sep
        )
        super().save_to_path(path)

    def sample(self, n: int) -> None:
        self.df = self.df.sample(n=n)

    @classmethod
    def config_preprocessors(cls) -> list:
        def preprocess_df(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            if "filename" not in config or config["filename"] is None:
                raise ValueError("Filename not specified in config")
            path = cls.local_path_from_name(name=config["name"], project=config["project"])
            filepath = os.path.join(path, config["filename"])
            if os.path.exists(filepath):
                if "df" in config and config["df"] is not None:
                    logger.warning("Dataframe NOT loaded from existing file")
                else:
                    # keep_deault_na set to False so empty strings stay empty strings
                    df = pd.read_csv(filepath, keep_default_na=False, sep=config["sep"])
                    config["df"] = df
            else:
                logger.warning(f"CSV filepath {filepath} does not exists")

        return super().config_preprocessors() + [preprocess_df]
