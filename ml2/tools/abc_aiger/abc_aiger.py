"""ABC and AIGER"""

import base64
import json
import logging
import re

from grpc._channel import _InactiveRpcError

from ...aiger import AIGERCircuit
from ...globals import CONTAINER_REGISTRY
from ...grpc import abc_aiger_pb2, abc_aiger_pb2_grpc
from ...grpc.aiger import aiger_pb2
from ...grpc.tools.tools_pb2 import SetupRequest
from ..grpc_service import GRPCService
from ..ltl_tool.pb2_converter import TimeConverterPb2
from .abc_aiger_grpc_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ABCAIGER_IMAGE_NAME = CONTAINER_REGISTRY + "/abc_aiger-grpc-server:latest"


class ABCAiger(GRPCService):

    def __init__(self, image: str = ABCAIGER_IMAGE_NAME, service=serve, **kwargs):
        super().__init__(
            stub=abc_aiger_pb2_grpc.ABCAigerStub,
            image=image,
            service=service,
            tool="ABCAiger",
            **kwargs
        )
        setup_response = self.stub.Setup(SetupRequest(parameters={}))
        if not setup_response.success:
            raise Exception(
                "Tool setup for " + self.tool + " failed. \n Error: " + setup_response.error
            )

    def convert_binary_to_aasci(
        self, circuit_bin: str, parameters: dict
    ) -> tuple[AIGERCircuit, str, str, TimeConverterPb2]:
        """
        Converts a binary circuit representation to an ASCII format.

        Args:
            circuit_bin (str): The binary representation of the circuit.
            parameters (dict): A dictionary of parameters, e.g timeout.

        Returns:
            tuple: A tuple containing:
                - AIGERCircuit: The converted AIGER circuit object.
                - str: An error message.
                - str: The tool name.
                - TimeConverterPb2: Time taken for the conversion.

        Raises:
           _InactiveRpcError: If the gRPC connection failed.
        """
        aig_pb2 = aiger_pb2.AigerBinaryCircuit(circuit=circuit_bin)
        result_pb2 = self.stub.ConvertBinaryToAasci(
            abc_aiger_pb2.ConvertBinaryToAasciRequest(
                aig=aig_pb2,
                parameters={k: json.dumps(v) for k, v in parameters.items()},
            )
        )
        return (
            AIGERCircuit.from_str(result_pb2.aag.circuit),
            result_pb2.error,
            result_pb2.tool,
            TimeConverterPb2(result_pb2.time),
        )

    def convert_aasci_to_binary(
        self, circuit: AIGERCircuit, parameters: dict
    ) -> tuple[str, str, str, TimeConverterPb2]:
        """
        Converts an AIGER circuit from ASCII format to binary format.

        Args:
            circuit (AIGERCircuit): The AIGER circuit to be converted.
            parameters (dict): A dictionary of parameters, e.g. timeout.

        Returns:
            tuple: A tuple containing:
            - str: The binary representation of the circuit.
            - str: An error message.
            - str: The tool name..
            - TimeConverterPb2: Time taken for the conversion.

        Raises:
           _InactiveRpcError: If the gRPC connection failed.
        """
        result_pb2 = self.stub.ConvertAasciToBinary(
            abc_aiger_pb2.ConvertAasciToBinaryRequest(
                aag=aiger_pb2.AigerCircuit(circuit=circuit.to_str()),
                parameters={k: json.dumps(v) for k, v in parameters.items()},
            )
        )
        return (
            result_pb2.aig.circuit,
            result_pb2.error,
            result_pb2.tool,
            TimeConverterPb2(result_pb2.time),
        )

    def convert_aiger_to_dot(
        self, circuit: AIGERCircuit, parameters: dict
    ) -> tuple[str, str, str, TimeConverterPb2]:
        """
        Converts an AIGER circuit from ASCII format to dot visualization.

        Args:
            circuit (AIGERCircuit): The AIGER circuit to be converted.
            parameters (dict): A dictionary of parameters, e.g. timeout.

        Returns:
            tuple: A tuple containing:
            - str: The dots string visualizing the circuit
            - str: An error message.
            - str: The tool name..
            - TimeConverterPb2: Time taken for the conversion.

        Raises:
           _InactiveRpcError: If the gRPC connection failed.
        """
        result_pb2 = self.stub.ConvertAigerToDot(
            abc_aiger_pb2.ConvertAigerToDotRequest(
                aag=aiger_pb2.AigerCircuit(circuit=circuit.to_str()),
                parameters={k: json.dumps(v) for k, v in parameters.items()},
            )
        )
        return (
            result_pb2.dot,
            result_pb2.error,
            result_pb2.tool,
            TimeConverterPb2(result_pb2.time),
        )

    # TODO test

    def convert_dot_to_png(self, dot: str) -> tuple[str, str, TimeConverterPb2]:
        """
        Converts a dot string to a PNG image (base_64 encoded).

        Args:
            dot (str): The dot string to be converted.

        Returns:
            str: The PNG image as a base64 encoded string.
        """
        result_pb2 = self.stub.ConvertDotToPng(abc_aiger_pb2.ConvertDotRequest(dot=dot))
        image_data = re.sub(r"^data:image/.+;base64,", "", result_pb2.png)
        image = base64.b64decode(image_data)
        return image, result_pb2.tool, TimeConverterPb2(result_pb2.time)

    def convert_dot_to_svg(self, dot: str) -> tuple[str, str, TimeConverterPb2]:
        """
        Converts a dot string to a SVG image.

        Args:
            dot (str): The dot string to be converted.

        Returns:
            str: The SVG image as a string.
        """
        result_pb2 = self.stub.ConvertDotToSvg(abc_aiger_pb2.ConvertDotRequest(dot=dot))
        return result_pb2.svg, result_pb2.tool, TimeConverterPb2(result_pb2.time)

    def display_dot(self, dot: str) -> None:
        """
        Displays a dot string as a PNG image.

        Args:
            dot (str): The dot string to be displayed.
        """
        from IPython.display import SVG, display

        png, _, _ = self.convert_dot_to_svg(dot)
        display(SVG(png))

    def display_aiger(self, circuit: AIGERCircuit, parameters: dict) -> None:
        """
        Displays the AIGER circuit through the DOT format.

        This method converts the given AIGER circuit to the DOT format and displays it.
        If the conversion fails, it prints an error message.

        Args:
            circuit (AIGERCircuit): The AIGER circuit to be displayed.
            parameters (dict): A dictionary of parameters for the conversion process.

        Returns:
            None
        """
        dot, error, _, _ = self.convert_aiger_to_dot(circuit, parameters)
        if dot is not None:
            self.display_dot(dot)
        else:
            print("Error: " + error)

    def aiger_simplify(
        self, circuit: AIGERCircuit, parameters: dict, abc_commands: list[str] | None = None
    ) -> tuple[list[AIGERCircuit], str, str, TimeConverterPb2]:
        """
        Simplifies an AIGER circuit. Uses the list of ABC commands to operate on the circuit.
        The right sequence of commands can lead to smaller circuits.
        Default are KNOR simplification commands.

        Args:
            circuit (AIGERCircuit): The AIGER circuit to be converted.
            parameters (dict): A dictionary of parameters, e.g. timeout. Timeout per abc and aiger call not for the whole operation.

        Returns:
            tuple: A tuple containing:
            - list[AIGERCircuit]: The history of AIGER circuits. Last circuits is the simplified version.
            - str: An error message.
            - str: The tool name.
            - TimeConverterPb2: Time taken for the conversion.

        Raises:
           _InactiveRpcError: If the gRPC connection failed.
        """
        result_pb2 = self.stub.AigerSimplify(
            abc_aiger_pb2.AigerSimplifyRequest(
                aag=aiger_pb2.AigerCircuit(circuit=circuit.to_str()),
                simplify_commands=abc_commands,
                parameters={k: json.dumps(v) for k, v in parameters.items()},
            )
        )
        return (
            [AIGERCircuit.from_str(circ.circuit) for circ in result_pb2.aag],
            result_pb2.error,
            result_pb2.tool,
            TimeConverterPb2(result_pb2.time),
        )
