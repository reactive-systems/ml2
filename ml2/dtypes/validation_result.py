"""Abstract validation result class"""

from abc import abstractmethod
from typing import Optional

from .csv_dtype import CSVLoggable
from .dtype import DType


class ValidationResult(DType):
    @property
    @abstractmethod
    def validation_success(self) -> Optional[bool]:
        """Return true if validiation was succesfull"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def validation_status(self) -> Optional[str]:
        """Return more detailed status of validation"""
        raise NotImplementedError()


class CSVLoggableValidationResult(ValidationResult, CSVLoggable):
    pass
