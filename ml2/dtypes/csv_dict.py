"""Simple dict data type that inherits from CSV data type"""


from typing import Dict, List

from ..registry import register_type
from .csv_dtype import CSV


@register_type
class CSVDict(CSV, dict):
    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        assert all(isinstance(key, str) for key in self)
        return {key: str(value) for key, value in self.items()}

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "CSVDict":
        return cls(fields)

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        raise NotImplementedError("Header not supported for generic dicts")
