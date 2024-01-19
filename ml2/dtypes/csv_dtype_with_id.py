"""Abstract CSV data type with ID class"""

from typing import Dict, List, Optional, Type, TypeVar

from .csv_dtype import CSV
from .hashable import Hashable

T = TypeVar("T", bound="CSVWithId")


class CSVWithId(CSV, Hashable):
    """Parent class for all data types that handle csv structures, i.e that need from_csv_fields and to_csv_fields functionality.
    Has support for identifiers.
    All child classes should implement the function _to_csv_fields() and _from_csv_fields()."""

    def __init__(
        self,
        unique_id: Optional[str] = None,
    ) -> None:
        Hashable.__init__(self, unique_id)

    def to_csv_fields(
        self,
        id_key: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        catch_unique_id_error: bool = True,
        **kwargs
    ) -> Dict[str, str]:
        """Writes selected fields to a dict that can then be used to populate a csv file.
        This is a wrapper to  _to_csv_fields(), which additionally writes the identifier of the class.

        Returns:
            Dict[str, str]: A dictionary containing the populated fields
        """
        prefix = "" if prefix is None else prefix
        suffix = "" if suffix is None else suffix
        key = (prefix + "id_" + self.__class__.__name__) + suffix if id_key is None else id_key
        unique_id = self.unique_id(catch_error=catch_unique_id_error)
        unique_id_dict = {key: str(unique_id)} if unique_id is not None else {}
        return {**super().to_csv_fields(prefix=prefix, suffix=suffix, **kwargs), **unique_id_dict}

    @classmethod
    def csv_field_header(
        cls,
        id_key: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Gives the header (keys) of the to_csv_field() method as class method.
        This is a stub of _csv_field_header().

        Returns:
            List[str]: A list containing the field keys
        """
        prefix = "" if prefix is None else prefix
        suffix = "" if suffix is None else suffix
        key = (prefix + "id_" + cls.__name__) + suffix if id_key is None else id_key
        return list(set(super().csv_field_header(prefix=prefix, suffix=suffix, **kwargs) + [key]))

    @classmethod
    def from_csv_fields(
        cls: Type[T],
        fields: Dict[str, str],
        id_key: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> T:
        """Reads a fields dict, from which the CSV data type can be constructed.
        This is a wrapper to  _from_csv_fields(), which additionally reads the identifier of the class.

        Args:
            fields (Dict[str, str]): A dictionary from which the class is constructed.

        Returns:
            CSV: The constructed CSV data type object
        """
        prefix = prefix if prefix is not None else ""
        suffix = suffix if suffix is not None else ""
        class_obj: T = super().from_csv_fields(
            fields=fields, prefix=prefix, suffix=suffix, **kwargs
        )
        key = ("id_" + class_obj.__class__.__name__) if id_key is None else id_key
        key = cls._remove_prefix_suffix(key, prefix=prefix, suffix=suffix)
        if key in fields.keys():
            class_obj._unique_id_value = fields[key]
        return class_obj
