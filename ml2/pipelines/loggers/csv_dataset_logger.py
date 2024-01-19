"""CSV Logger class"""


from typing import Any

from ...datasets import ContCSVDatasetWriter
from ...dtypes import CSVDict
from ...registry import register_type
from ..samples import Sample
from .csv_logger import CSVLogger


@register_type
class CSVToDatasetLogger(ContCSVDatasetWriter[CSVDict], CSVLogger):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, dtype=CSVDict, **kwargs)

    def add(self, sample: Sample, **kwargs) -> Any:
        for fields in self.process_generic_sample(sample, **kwargs):
            self.add_sample(CSVDict(fields))
