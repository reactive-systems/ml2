"""Tool LTL conversion"""

import json
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from ...grpc.syfco import syfco_pb2
from ...ltl.ltl_spec import DecompLTLSpec
from .pb2_converter import SpecificationConverterPb2, TimeConverterPb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolLTLConversionRequest:
    def __init__(self, tlsf_string: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.parameters = parameters if parameters is not None else {}
        self.tlsf_string = tlsf_string

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolLTLConversionRequest):
            return False
        else:
            return self.parameters == __o.parameters and self.tlsf_string == __o.tlsf_string

    def to_pb2_ConvertTLSFToSpecRequest(self, **kwargs) -> syfco_pb2.ConvertTLSFToSpecRequest:
        params = {k: json.dumps(v) for k, v in self.parameters.items()}
        return syfco_pb2.ConvertTLSFToSpecRequest(
            parameters=params, tlsf=syfco_pb2.TLSFFileString(tlsf=self.tlsf_string)
        )

    @classmethod
    def from_pb2_ConvertTLSFToSpecRequest(
        cls, pb2_obj: syfco_pb2.ConvertTLSFToSpecRequest, **kwargs
    ) -> "ToolLTLConversionRequest":
        parameters = {k: json.loads(v) for k, v in pb2_obj.parameters.items()}
        tlsf_string = pb2_obj.tlsf.tlsf

        return cls(parameters=parameters, tlsf_string=tlsf_string)


class ToolLTLConversionResponse(TimeConverterPb2, SpecificationConverterPb2):
    def __init__(
        self,
        error: str,
        time: timedelta,
        tool: str,
        specification: Optional[DecompLTLSpec] = None,
    ) -> None:
        self.error = error
        self.tool = tool
        TimeConverterPb2.__init__(self, time)
        SpecificationConverterPb2.__init__(self, decomp_specification=specification)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolLTLConversionResponse):
            return False
        else:
            return (
                self.error == __o.error
                and self.tool == __o.tool
                and TimeConverterPb2.__eq__(self, __o)
                and SpecificationConverterPb2.__eq__(self, __o)
            )

    def to_pb2_ConvertTLSFToSpecResponse(self, **kwargs) -> syfco_pb2.ConvertTLSFToSpecResponse:
        time = self.time_to_pb2(**kwargs)
        specification = (
            {"specification": self.specification_to_pb2(**kwargs)["decomp_specification"]}
            if self.decomp_specification is not None
            else {}
        )

        return syfco_pb2.ConvertTLSFToSpecResponse(
            **specification,
            error=self.error,
            tool=self.tool,
            time=time,
        )

    @classmethod
    def from_pb2_ConvertTLSFToSpecResponse(
        cls, pb2_obj: syfco_pb2.ConvertTLSFToSpecResponse, **kwargs
    ) -> "ToolLTLConversionResponse":
        error: str = pb2_obj.error
        tool: str = pb2_obj.tool
        time: timedelta = cls.from_time_tb2(pb2_obj.time, **kwargs)
        specification = (
            cls.from_specification_pb2(
                decomp_specification_pb2=pb2_obj.specification, formula_specification_pb2=None
            )[0]
            if pb2_obj.HasField("specification")
            else None
        )

        return cls(
            specification=specification,
            error=error,
            tool=tool,
            time=time,
        )
