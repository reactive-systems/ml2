"""Abstract Logger class"""

from typing import Any, Generic, List, TypeVar

from ...dtypes import DType
from ...dtypes.validation_result import ValidationResult
from ..callbacks import Callback
from ..samples import (
    Beam,
    BeamSearchLabeledSample,
    BeamSearchSample,
    EncodedSample,
    EvalLabeledSample,
    EvalSample,
    LabeledSample,
    PortfolioSample,
    Result,
    Sample,
    VerifiedBeam,
    VerifiedBeamSearchLabeledSample,
    VerifiedBeamSearchSample,
    VerifiedLabeledSample,
    VerifiedSample,
)

R = TypeVar("R")
D = TypeVar("D", bound=DType)


class SampleLogger(Callback, Generic[R]):
    def process_generic_sample(self, sample: Sample[Any], **kwargs) -> List[R]:
        if isinstance(sample, VerifiedBeamSearchLabeledSample):
            return self.process_verified_beam_search_supervised_sample(sample, **kwargs)
        elif isinstance(sample, VerifiedLabeledSample):
            return self.process_verified_supervised_sample(sample, **kwargs)
        elif isinstance(sample, VerifiedBeamSearchSample):
            return self.process_verified_beam_search_sample(sample, **kwargs)
        elif isinstance(sample, VerifiedSample):
            return self.process_verified_sample(sample, **kwargs)
        elif isinstance(sample, BeamSearchLabeledSample):
            return self.process_beam_search_supervised_sample(sample, **kwargs)
        elif isinstance(sample, EvalLabeledSample):
            return self.process_eval_supervised_sample(sample, **kwargs)
        elif isinstance(sample, LabeledSample):
            return self.process_supervised_sample(sample, **kwargs)
        elif isinstance(sample, BeamSearchSample):
            return self.process_beam_search_sample(sample, **kwargs)
        elif isinstance(sample, EvalSample):
            return self.process_eval_sample(sample, **kwargs)
        elif isinstance(sample, EncodedSample):
            return self.process_encoded_sample(sample, **kwargs)
        elif isinstance(sample, PortfolioSample):
            return self.process_portfolio_sample(sample, **kwargs)
        elif isinstance(sample, Sample):
            return self.process_sample(sample, **kwargs)
        else:
            raise TypeError

    @classmethod
    def process_portfolio_result(cls, result: Result[Any], **kwargs) -> R:
        raise NotImplementedError

    @classmethod
    def process_portfolio_sample(cls, sample: PortfolioSample[Any, Any], **kwargs) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_beam(cls, beam: Beam[Any], **kwargs) -> R:
        raise NotImplementedError

    @classmethod
    def process_verified_beam(cls, beam: VerifiedBeam[Any, ValidationResult], **kwargs) -> R:
        raise NotImplementedError

    @classmethod
    def process_encoded_sample(cls, sample: EncodedSample[Any], **kwargs) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_sample(cls, sample: Sample[Any], **kwargs) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_eval_sample(cls, sample: EvalSample[Any, Any], **kwargs) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_beam_search_sample(cls, sample: BeamSearchSample[Any, Any], **kwargs) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_supervised_sample(cls, sample: LabeledSample[Any, Any], **kwargs) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_eval_supervised_sample(
        cls, sample: EvalLabeledSample[Any, Any], **kwargs
    ) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_beam_search_supervised_sample(
        cls, sample: BeamSearchLabeledSample[Any, Any], **kwargs
    ) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_verified_sample(
        cls, sample: VerifiedSample[Any, Any, ValidationResult], **kwargs
    ) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_verified_beam_search_sample(
        cls, sample: VerifiedBeamSearchSample[Any, Any, ValidationResult], **kwargs
    ) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_verified_supervised_sample(
        cls, sample: VerifiedLabeledSample[Any, Any, ValidationResult], **kwargs
    ) -> List[R]:
        raise NotImplementedError

    @classmethod
    def process_verified_beam_search_supervised_sample(
        cls, sample: VerifiedBeamSearchLabeledSample[Any, Any, ValidationResult], **kwargs
    ) -> List[R]:
        raise NotImplementedError
