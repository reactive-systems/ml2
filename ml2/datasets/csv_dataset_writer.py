"""CSV dataset writer"""

import csv
import os
from typing import Any, Dict, List, Type, TypeVar

from ..dtypes import CSV
from .dataset_writer import DatasetWriter

T = TypeVar("T", bound=CSV)


class CSVDatasetWriter(DatasetWriter[T]):
    def __init__(
        self,
        name: str,
        dtype: Type[T],
        header: List[str],
        filename: str = None,
        sep: str = ",",
        **kwargs,
    ):
        self.dtype = dtype
        self.header = header
        self._filename = filename
        self.sep = sep

        super().__init__(name=name, **kwargs)
        filepath = os.path.join(self.local_path, self.filename)
        if os.path.exists(filepath):
            raise Exception(f"Filepath {filepath} already exists")
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)
        self.file = open(filepath, "x", newline="")
        self.file_writer = csv.DictWriter(
            self.file, fieldnames=self.header, delimiter=self.sep, quoting=csv.QUOTE_ALL
        )
        self.file_writer.writeheader()
        self._write_counter = 0

    def add_sample(self, sample: T, **kwargs) -> None:
        csv_fields = sample.to_csv_fields(**kwargs)
        self.file_writer.writerow({key: csv_fields.get(key, None) for key in self.header})
        self._write_counter += 1

    def close(self) -> None:
        self.file.close()

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_header(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("header", None)
            annotations.pop("header", None)

        def postprocess_type(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config["type"] = "CSVDataset"

        return [postprocess_header, postprocess_type] + super().config_postprocessors()

    @property
    def filename(self) -> str:
        return self._filename if self._filename is not None else "data.csv"

    def size(self, **kwargs) -> int:
        return self._write_counter


class ContCSVDatasetWriter(DatasetWriter[T]):
    def __init__(
        self,
        name: str,
        dtype: Type[T],
        filename: str = None,
        sep: str = ",",
        **kwargs,
    ):
        self.dtype = dtype
        self.header = []
        self._filename = filename
        self.sep = sep
        self.rows: List[Dict[str, str]] = []
        self._file = None
        self._file_writer = None

        super().__init__(name=name, **kwargs)
        filepath = os.path.join(self.local_path, self.filename)
        if os.path.exists(filepath):
            raise Exception(f"Filepath {filepath} already exists")
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)
        self.filepath = filepath
        self._write_counter = 0

    @property
    def file(self):
        if self._file is None:
            self._file = open(self.filepath, "x", newline="")
        return self._file

    @property
    def file_writer(self):
        if self._file_writer is None:
            self._file_writer = csv.DictWriter(
                self.file, fieldnames=self.header, delimiter=self.sep, quoting=csv.QUOTE_ALL
            )
            self._file_writer.writeheader()
        return self._file_writer

    def update_header(self, header: List[str]):
        if not set(header).issubset(set(self.header)):
            print("keys changed -- rewriting data...")
            self.header = list(set(header).union(set(self.header)))
            self.rewrite()

    def rewrite(self):
        self.file.close()
        self._file = None
        self._file_writer = None
        os.remove(self.filepath)
        for row in self.rows:
            self.file_writer.writerow(row)

    def add_sample(self, sample: T, **kwargs) -> None:
        csv_fields = sample.to_csv_fields(**kwargs)
        self.update_header(list(csv_fields.keys()))
        self.file_writer.writerow(
            {key: (csv_fields[key] if key in csv_fields else "") for key in self.header}
        )
        self.file.flush()
        self.rows.append(csv_fields)
        self._write_counter += 1

    def close(self) -> None:
        self.file.close()

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_type(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config["type"] = "CSVDataset"

        return [postprocess_type] + super().config_postprocessors()

    @property
    def filename(self) -> str:
        return self._filename if self._filename is not None else "data.csv"

    def size(self, **kwargs) -> int:
        return self._write_counter
