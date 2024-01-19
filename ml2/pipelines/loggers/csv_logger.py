"""CSV Logger"""

from typing import Dict, List

from ...datasets.utils import to_csv_str
from ...dtypes import CSVLoggable, CSVLoggableValidationResult
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
from .sample_logger import SampleLogger


class CSVLogger(SampleLogger[Dict[str, str]]):
    @classmethod
    def process_portfolio_result(cls, result: Result[CSVLoggable], **kwargs) -> Dict[str, str]:
        id = {"result_id": str(result.id)} if result.id is not None else {}
        name: dict[str, str] = {"result_name": result.name} if result.name is not None else {}
        result_d = result.result.to_csv_fields(prefix="result_", **kwargs) if result.result else {}
        time = {"result_time": str(result.time)} if result.time is not None else {}
        return {**id, **name, **result_d, **time}

    @classmethod
    def process_portfolio_sample(
        cls, sample: PortfolioSample[CSVLoggable, CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        sample_dict = cls.process_sample(sample, **kwargs)[0]

        if len(sample.results) == 0:
            return [sample_dict]
        else:
            sample_list = []
            for result in sample.results:
                sample_list.append(
                    {
                        **sample_dict,
                        **cls.process_portfolio_result(result, **kwargs),
                    }
                )
            return sample_list

    @classmethod
    def process_beam(cls, beam: Beam[CSVLoggable], **kwargs) -> Dict[str, str]:
        prediction = beam.pred.to_csv_fields(prefix="prediction_", **kwargs) if beam.pred else {}
        pred_enc = (
            {"prediction_enc": to_csv_str(str(beam.pred_enc))} if beam.pred_enc is not None else {}
        )
        pred_dec_err = (
            {"prediction_dec_err": to_csv_str(str(beam.pred_dec_err))}
            if beam.pred_dec_err is not None
            else {}
        )
        id = {"id": str(beam.id)} if beam.id is not None else {}
        syn_time = {"syn_time": str(beam.time)} if beam.time else {}

        return {**prediction, **pred_enc, **pred_dec_err, **id, **syn_time}

    @classmethod
    def process_verified_beam(
        cls, beam: VerifiedBeam[CSVLoggable, CSVLoggableValidationResult], **kwargs
    ) -> Dict[str, str]:
        verification = (
            beam.verification.to_csv_fields(**kwargs, prefix="verification_")
            if beam.verification is not None
            else {}
        )
        verification_err = (
            {"verification_err": to_csv_str(beam.verification_err)}
            if beam.verification_err is not None
            else {}
        )
        ver_time = {"ver_time": str(beam.verification_time)} if beam.verification_time else {}

        return {
            **cls.process_beam(beam, **kwargs),
            **verification,
            **verification_err,
            **ver_time,
        }

    @classmethod
    def process_encoded_sample(
        cls, sample: EncodedSample[CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        inp_enc = (
            {"input_enc": to_csv_str(str(sample.inp_enc))} if sample.inp_enc is not None else {}
        )
        inp_enc_err = (
            {"input_enc_err": to_csv_str(str(sample.inp_enc_err))}
            if sample.inp_enc_err is not None
            else {}
        )
        return [
            {
                **cls.process_sample(sample, **kwargs)[0],
                **inp_enc,
                **inp_enc_err,
            }
        ]

    @classmethod
    def process_sample(cls, sample: Sample[CSVLoggable], **kwargs) -> List[Dict[str, str]]:
        input: Dict[str, str] = sample.inp.to_csv_fields(prefix="input_", **kwargs)
        id = {"id": str(sample.id)} if sample.id is not None else {}
        name: dict[str, str] = {"name": sample.name} if sample.name is not None else {}

        return [
            {
                **input,
                **id,
                **name,
            }
        ]

    @classmethod
    def process_eval_sample(
        cls, sample: EvalSample[CSVLoggable, CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        prediction = (
            sample.pred.to_csv_fields(**kwargs, prefix="prediction_")
            if sample.pred is not None
            else {}
        )
        pred_enc = (
            {"prediction_enc": to_csv_str(str(sample.pred_enc))}
            if sample.pred_enc is not None
            else {}
        )
        pred_dec_err = (
            {"prediction_dec_err": to_csv_str(str(sample.pred_dec_err))}
            if sample.pred_dec_err is not None
            else {}
        )
        syn_time = {"syn_time": str(sample.time)} if sample.time else {}

        return [
            {
                **cls.process_encoded_sample(sample, **kwargs)[0],
                **prediction,
                **pred_enc,
                **pred_dec_err,
                **syn_time,
            }
        ]

    @classmethod
    def process_beam_search_sample(
        cls, sample: BeamSearchSample[CSVLoggable, CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        sample_dict = cls.process_eval_sample(sample, **kwargs)[0]
        if len(sample.beams) == 0:
            return [sample_dict]
        else:
            sample_list = []
            for beam in sample.beams:
                sample_list.append({**sample_dict, **cls.process_beam(beam, **kwargs)})
            return sample_list

    @classmethod
    def process_supervised_sample(
        cls, sample: LabeledSample[CSVLoggable, CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        target = sample.tar.to_csv_fields(**kwargs, prefix="target_")
        tar_enc = (
            {"target_enc": to_csv_str(str(sample.tar_enc))} if sample.tar_enc is not None else {}
        )
        tar_enc_err = (
            {"target_enc_err": to_csv_str(str(sample.tar_enc_err))}
            if sample.tar_enc_err is not None
            else {}
        )
        return [
            {
                **cls.process_encoded_sample(sample, **kwargs)[0],
                **target,
                **tar_enc,
                **tar_enc_err,
            }
        ]

    @classmethod
    def process_eval_supervised_sample(
        cls, sample: EvalLabeledSample[CSVLoggable, CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        return [
            {
                **cls.process_eval_sample(sample, **kwargs)[0],
                **cls.process_supervised_sample(sample, **kwargs)[0],
            }
        ]

    @classmethod
    def process_beam_search_supervised_sample(
        cls, sample: BeamSearchLabeledSample[CSVLoggable, CSVLoggable], **kwargs
    ) -> List[Dict[str, str]]:
        sample_dict = cls.process_eval_supervised_sample(sample, **kwargs)[0]
        if len(sample.beams) == 0:
            return [sample_dict]
        else:
            sample_list = []
            for beam in sample.beams:
                sample_list.append({**sample_dict, **cls.process_beam(beam, **kwargs)})
            return sample_list

    @classmethod
    def process_verified_sample(
        cls,
        sample: VerifiedSample[CSVLoggable, CSVLoggable, CSVLoggableValidationResult],
        **kwargs
    ) -> List[Dict[str, str]]:
        verification = (
            sample.verification.to_csv_fields(**kwargs, prefix="verification_")
            if sample.verification is not None
            else {}
        )
        verification_err = (
            {"verification_err": to_csv_str(sample.verification_err)}
            if sample.verification_err is not None
            else {}
        )
        ver_time = {"ver_time": str(sample.verification_time)} if sample.verification_time else {}
        return [
            {
                **cls.process_eval_sample(sample, **kwargs)[0],
                **verification_err,
                **verification,
                **ver_time,
            }
        ]

    @classmethod
    def process_verified_beam_search_sample(
        cls,
        sample: VerifiedBeamSearchSample[CSVLoggable, CSVLoggable, CSVLoggableValidationResult],
        **kwargs
    ) -> List[Dict[str, str]]:
        sample_dict = cls.process_encoded_sample(sample, **kwargs)[0]
        if len(sample.beams) == 0:
            return [sample_dict]
        else:
            sample_list = []
            for beam in sample.beams:
                sample_list.append({**sample_dict, **cls.process_verified_beam(beam, **kwargs)})
            return sample_list

    @classmethod
    def process_verified_supervised_sample(
        cls,
        sample: VerifiedLabeledSample[CSVLoggable, CSVLoggable, CSVLoggableValidationResult],
        **kwargs
    ) -> List[Dict[str, str]]:
        verification = (
            sample.verification.to_csv_fields(**kwargs, prefix="verification_")
            if sample.verification is not None
            else {}
        )
        verification_err = (
            {"verification_err": sample.verification_err}
            if sample.verification_err is not None
            else {}
        )
        ver_time = {"ver_time": str(sample.verification_time)} if sample.verification_time else {}
        return [
            {
                **cls.process_eval_supervised_sample(sample, **kwargs)[0],
                **verification_err,
                **verification,
                **ver_time,
            }
        ]

    @classmethod
    def process_verified_beam_search_supervised_sample(
        cls,
        sample: VerifiedBeamSearchLabeledSample[
            CSVLoggable, CSVLoggable, CSVLoggableValidationResult
        ],
        **kwargs
    ) -> List[Dict[str, str]]:
        sample_dict = cls.process_supervised_sample(sample, **kwargs)[0]
        sample_list = []
        for beam in sample.beams:
            sample_list.append({**sample_dict, **cls.process_verified_beam(beam, **kwargs)})
        return sample_list
