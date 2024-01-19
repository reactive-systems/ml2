"""Abstract CSV data type class"""

from abc import abstractmethod
from typing import Dict, List, Optional, Type, TypeVar

from .dtype import DType

T = TypeVar("T", bound="CSV")


class CSVLoggable(DType):
    """Parent class for all DataTypes that can save to csv structures, i.e that need to_csv_fields functionality.
    All child classes should implement the function _to_csv_fields()."""

    def to_csv_fields(
        self, prefix: Optional[str] = None, suffix: Optional[str] = None, **kwargs
    ) -> Dict[str, str]:
        """Writes selected fields to a dict that can then be used to populate a csv file.
        This is a stub of _to_csv_fields().

        Returns:
            Dict[str, str]: A dictionary containing the populated fields
        """
        prefix = "" if prefix is None else prefix
        suffix = "" if suffix is None else suffix
        dic = self._to_csv_fields(**kwargs)
        return {prefix + d + suffix: dic[d] for d in dic}

    @classmethod
    def csv_field_header(
        cls, prefix: Optional[str] = None, suffix: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Gives the header (keys) of the to_csv_field() method as class method.
        This is a stub of _csv_field_header().

        Returns:
            List[str]: A list containing the field keys
        """
        prefix = "" if prefix is None else prefix
        suffix = "" if suffix is None else suffix
        lis = cls._csv_field_header(**kwargs)
        return [prefix + e + suffix for e in lis]

    @abstractmethod
    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        """Writes selected fields to a dict that can then be used to populate a csv file.
        Each children class should implement this function, otherwise the dictionary will be empty.
        Is wrapped by to_csv_fields(), to add additional functionality.
        Should not be called at any point, instead use to_csv_fields().

        Returns:
            Dict[str, str]:  A dictionary containing the populated fields
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        """Gives the header (keys) of the _to_csv_field() method as static method.
        Each children class should implement this function.
        Is wrapped by csv_field_header(), to add additional functionality.
        Should not be called at any point, instead use csv_field_header().

        Returns:
            List[str]:  A List containing the field keys
        """
        raise NotImplementedError()


class CSV(CSVLoggable):
    """Parent class for all data types that handle csv structures, i.e that need from_csv_fields and to_csv_fields functionality.
    All child classes should implement the function _to_csv_fields() and _from_csv_fields().
    Generally speaking, the following should hold: x == from_csv_fields(to_csv_fields(x))"""

    @classmethod
    def from_csv_fields(
        cls: Type[T],
        fields: Dict[str, str],
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> T:
        """Reads a fields dict, from which the CSV data type can be constructed.
        This is a stub of _from_csv_fields(), which additionally reads the identifier of the class.

        Args:
            fields (Dict[str, str]): A dictionary from which the class is constructed.

        Returns:
            CSV: The constructed CSV data type object
        """

        prefix = prefix if prefix is not None else ""
        suffix = suffix if suffix is not None else ""

        fields = {cls._remove_prefix_suffix(d, prefix, suffix): fields[d] for d in fields}
        return cls._from_csv_fields(fields=fields, **kwargs)

    @staticmethod
    def _remove_prefix_suffix(key: str, prefix: str, suffix: str) -> str:
        if key.startswith(prefix):
            key = key[len(prefix) :]
        if key.endswith(suffix):
            key = key[: (-len(suffix)) if len(suffix) != 0 else len(key)]
        return key

    @classmethod
    @abstractmethod
    def _from_csv_fields(cls: Type[T], fields: Dict[str, str], **kwargs) -> T:
        """Reads a fields dict, from which the CSV data type can be constructed.
        Each children class should implement this function, otherwise it will construct an empty class.
        Is wrapped by from_csv_fields(), to add additional functionality.
        Should not be called at any point, instead use from_csv_fields().

        Args:
            fields (Dict[str, str]): A dictionary from which the class is constructed.

        Returns:
            CSV: The constructed CSV data type object
        """
        raise NotImplementedError()

    @classmethod
    def from_csv_row_and_header(cls, row: List[str], header: List[str], **kwargs) -> "CSV":
        if len(row) != len(header):
            raise ValueError("Row length does not equal header length")
        return cls.from_csv_fields({header[i]: row[i] for i in range(len(row))}, **kwargs)
