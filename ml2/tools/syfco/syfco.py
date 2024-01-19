"""SyFCo"""

import logging
import ntpath
from typing import Optional

from ...globals import CONTAINER_REGISTRY
from ...grpc.syfco import syfco_pb2_grpc
from ...grpc.tools import tools_pb2
from ...ltl import DecompLTLSpec
from ..grpc_service import GRPCService
from ..ltl_tool.tool_ltl_conversion import ToolLTLConversionRequest, ToolLTLConversionResponse
from .syfco_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYFCO_IMAGE_NAME = CONTAINER_REGISTRY + "/syfco-grpc-server:latest"


class Syfco(GRPCService):
    def __init__(self, image: str = SYFCO_IMAGE_NAME, service=serve, cpu_count: int = 1, **kwargs):
        super().__init__(
            stub=syfco_pb2_grpc.SyfcoStub,
            image=image,
            service=service,
            cpu_count=cpu_count,
            tool="SyFCo",
            **kwargs
        )
        setup_response = self.stub.Setup(tools_pb2.SetupRequest(parameters={}))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )

    def from_tlsf_str(
        self,
        tlsf_spec: str,
        name: Optional[str] = None,
    ) -> DecompLTLSpec:
        request = ToolLTLConversionRequest(
            parameters={}, tlsf_string=tlsf_spec
        ).to_pb2_ConvertTLSFToSpecRequest()
        response = ToolLTLConversionResponse.from_pb2_ConvertTLSFToSpecResponse(
            self.stub.ConvertTLSFToSpec(request)
        )
        if response.decomp_specification is not None:
            response.decomp_specification.name = name
            return response.decomp_specification
        else:
            raise Exception("file could not be converted: " + str(response.error))

    def from_tlsf_file(
        self,
        tlsf_file: str,
        name: Optional[str] = None,
    ) -> DecompLTLSpec:
        with open(tlsf_file, "r") as file:
            file_str = str(file.read())
        return self.from_tlsf_str(
            tlsf_spec=file_str,
            name=(
                ntpath.basename(tlsf_file)[:-5]
                if ntpath.basename(tlsf_file).endswith(".tlsf")
                else ntpath.basename(tlsf_file)
            )
            if name is None
            else name,
        )
