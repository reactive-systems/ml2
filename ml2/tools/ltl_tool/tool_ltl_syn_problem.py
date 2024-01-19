"""Tool LTL synthesis problem"""

import json
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from ...aiger import AIGERCircuit
from ...dtypes import CSVLoggable
from ...grpc.ltl import ltl_syn_pb2
from ...grpc.system import system_pb2
from ...ltl.ltl_spec import DecompLTLSpec, LTLSpec
from ...ltl.ltl_syn import LTLSynStatus
from ...mealy import MealyMachine
from .pb2_converter import SpecificationConverterPb2, SystemConverterPb2, TimeConverterPb2
from .tool_ltl_mc_problem import ToolLTLMCSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolLTLSynProblem(CSVLoggable, SpecificationConverterPb2):
    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        formula_specification: Optional[LTLSpec] = None,
        decomp_specification: Optional[DecompLTLSpec] = None,
        specification: Optional[Union[LTLSpec, DecompLTLSpec]] = None,
        system_format=None,
    ) -> None:
        self.parameters = parameters if parameters is not None else {}
        SpecificationConverterPb2.__init__(
            self, formula_specification, decomp_specification, specification
        )
        if self.formula_specification is None and self.decomp_specification is None:
            raise ValueError(
                "At least one of formula_specification or decomp_specification needs to be set."
            )
        self.system_format = system_format if system_format is not None else "aiger"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolLTLSynProblem):
            return False
        else:
            return self.parameters == __o.parameters and SpecificationConverterPb2.__eq__(
                self, __o
            )

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        params = {"parameters": json.dumps(self.parameters)}
        system_format = {"system_format": self.system_format}
        spec = self.specification_to_csv_fields(**kwargs)

        return {**params, **system_format, **spec}

    def to_pb2_LTLSynProblem(self, **kwargs) -> ltl_syn_pb2.LTLSynProblem:
        system_format = (
            {"system_format": system_pb2.SYSTEM_AIGER}
            if self.system_format == "aiger"
            else {"system_format": system_pb2.SYSTEM_MEALY}
            if self.system_format == "mealy"
            else {}
        )
        params = {k: json.dumps(v) for k, v in self.parameters.items()}
        return ltl_syn_pb2.LTLSynProblem(
            parameters=params,
            **system_format,
            **self.specification_to_pb2(**kwargs),
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["parameters", "system_format"] + cls.specification_csv_field_header(**kwargs)

    @classmethod
    def from_pb2_LTLSynProblem(
        cls, pb2_obj: ltl_syn_pb2.LTLSynProblem, **kwargs
    ) -> "ToolLTLSynProblem":
        decomp_specification, formula_specification = cls.from_specification_pb2(
            pb2_obj.decomp_specification if pb2_obj.HasField("decomp_specification") else None,
            pb2_obj.formula_specification if pb2_obj.HasField("formula_specification") else None,
            **kwargs,
        )

        parameters = {k: json.loads(v) for k, v in pb2_obj.parameters.items()}
        system_format = (
            "aiger"
            if pb2_obj.system_format == system_pb2.SYSTEM_AIGER
            else "mealy"
            if pb2_obj.system_format == system_pb2.SYSTEM_MEALY
            else None
        )

        return cls(
            parameters=parameters,
            decomp_specification=decomp_specification,
            formula_specification=formula_specification,
            system_format=system_format,
        )


class ToolLTLSynSolution(
    CSVLoggable,
    SystemConverterPb2,
    TimeConverterPb2,
):
    def __init__(
        self,
        status: LTLSynStatus,
        detailed_status: str,
        tool: str,
        time: timedelta,
        circuit: Optional[AIGERCircuit] = None,
        mealy_machine: Optional[MealyMachine] = None,
        realizable: Optional[bool] = None,
    ) -> None:
        self.status = status
        self.detailed_status = detailed_status
        self.tool = tool
        SystemConverterPb2.__init__(self, circuit, mealy_machine, realizable)
        TimeConverterPb2.__init__(self, time)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolLTLSynSolution):
            return False
        else:
            return (
                self.detailed_status == __o.detailed_status
                and self.status == __o.status
                and self.tool == __o.tool
                and SystemConverterPb2.__eq__(self, __o)
                and TimeConverterPb2.__eq__(self, __o)
            )

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        system = self.system_to_csv_fields(**kwargs)
        realizable = {"realizable": str(self.realizable)} if self.realizable is not None else {}
        status = self.status.to_csv_fields()
        detailed_status = (
            {"detailed_status": self.detailed_status} if self.detailed_status != "" else {}
        )
        tool = {"tool": self.tool}
        time = self.time_to_csv_fields(**kwargs)

        return {**system, **realizable, **status, **detailed_status, **tool, **time}

    def to_pb2_LTLSynSolution(self, **kwargs) -> ltl_syn_pb2.LTLSynSolution:
        if self.mealy_machine is not None and self.circuit is not None:
            raise ValueError("Only one of mealy_machine or circuit can be set.")
        system = self.system_to_pb2(**kwargs)
        status, detailed_status = self.status.to_pb2_LTLSynStatus(**kwargs)
        detailed_status = self.detailed_status if self.detailed_status != "" else detailed_status
        time = self.time_to_pb2(**kwargs)
        realizable = {"realizable": self.realizable} if self.realizable is not None else {}

        return ltl_syn_pb2.LTLSynSolution(
            **system,
            **realizable,
            status=status,
            detailed_status=detailed_status,
            tool=self.tool,
            time=time,
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return (
            ["realizable", "detailed_status"]
            + LTLSynStatus.csv_field_header(**kwargs)
            + cls.system_csv_field_header(**kwargs)
            + ["tool"]
            + cls.time_csv_field_header(**kwargs)
        )

    @classmethod
    def from_pb2_LTLSynSolution(
        cls, pb2_obj: ltl_syn_pb2.LTLSynSolution, **kwargs
    ) -> "ToolLTLSynSolution":
        status: LTLSynStatus = LTLSynStatus.from_pb2_LTLSynStatus(
            pb2_obj.status, pb2_obj.detailed_status, **kwargs
        )
        detailed_status: str = pb2_obj.detailed_status
        realizable: Optional[bool] = (
            None if not pb2_obj.HasField("realizable") else pb2_obj.realizable
        )
        tool: str = pb2_obj.tool
        time: timedelta = cls.from_time_tb2(pb2_obj.time, **kwargs)
        circuit, mealy_machine = cls.from_system_pb2(
            pb2_obj.circuit if pb2_obj.HasField("circuit") else None,
            pb2_obj.mealy_machine if pb2_obj.HasField("mealy_machine") else None,
            **kwargs,
        )

        return cls(
            circuit=circuit,
            mealy_machine=mealy_machine,
            status=status,
            detailed_status=detailed_status,
            realizable=realizable,
            tool=tool,
            time=time,
        )


class ToolNeuralLTLSynSolution(
    CSVLoggable,
    TimeConverterPb2,
):
    def __init__(
        self,
        synthesis_solution: ToolLTLSynSolution,
        tool: str,
        time: timedelta,
        model_checking_solution: Optional[ToolLTLMCSolution] = None,
    ) -> None:
        self.model_checking_solution = model_checking_solution
        self.synthesis_solution = synthesis_solution
        self.tool = tool
        TimeConverterPb2.__init__(self, time)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolNeuralLTLSynSolution):
            return False
        else:
            return (
                TimeConverterPb2.__eq__(self, __o)
                and self.tool == __o.tool
                and self.model_checking_solution == __o.model_checking_solution
                and self.synthesis_solution == __o.synthesis_solution
            )

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        tool = {"tool": self.tool}
        time = self.time_to_csv_fields(**kwargs)
        mc_solution = (
            self.model_checking_solution.to_csv_fields(prefix="model_checking_", **kwargs)
            if self.model_checking_solution is not None
            else {}
        )
        syn_solution = self.synthesis_solution.to_csv_fields(prefix="synthesis_", **kwargs)

        return {**syn_solution, **mc_solution, **tool, **time}

    def to_pb2_NeuralLTLSynSolution(self, **kwargs) -> ltl_syn_pb2.NeuralLTLSynSolution:
        time = self.time_to_pb2(**kwargs)
        model_checking_solution = (
            {
                "model_checking_solution": self.model_checking_solution.to_pb2_LTLMCSolution(
                    **kwargs
                )
            }
            if self.model_checking_solution is not None
            else {}
        )

        return ltl_syn_pb2.NeuralLTLSynSolution(
            **model_checking_solution,
            synthesis_solution=self.synthesis_solution.to_pb2_LTLSynSolution(),
            tool=self.tool,
            time=time,
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return (
            ToolLTLSynSolution.csv_field_header(prefix="synthesis_", **kwargs)
            + ToolLTLMCSolution.csv_field_header(prefix="model_checking_", **kwargs)
            + ["tool"]
            + cls.time_csv_field_header(**kwargs)
        )

    @classmethod
    def from_pb2_NeuralLTLSynSolution(
        cls, pb2_obj: ltl_syn_pb2.NeuralLTLSynSolution, **kwargs
    ) -> "ToolNeuralLTLSynSolution":
        tool: str = pb2_obj.tool
        time: timedelta = cls.from_time_tb2(pb2_obj.time, **kwargs)
        synthesis_solution = ToolLTLSynSolution.from_pb2_LTLSynSolution(pb2_obj.synthesis_solution)
        model_checking_solution = (
            ToolLTLMCSolution.from_pb2_LTLMCSolution(pb2_obj.model_checking_solution, **kwargs)
            if pb2_obj.HasField("model_checking_solution")
            else None
        )

        return cls(
            tool=tool,
            time=time,
            model_checking_solution=model_checking_solution,
            synthesis_solution=synthesis_solution,
        )


class ToolNeuralLTLSynSolutionSpecPair(CSVLoggable, SpecificationConverterPb2):
    def __init__(
        self,
        solution: ToolNeuralLTLSynSolution,
        formula_specification: Optional[LTLSpec] = None,
        decomp_specification: Optional[DecompLTLSpec] = None,
        specification: Optional[Union[LTLSpec, DecompLTLSpec]] = None,
    ) -> None:
        self.solution = solution
        SpecificationConverterPb2.__init__(
            self,
            formula_specification=formula_specification,
            decomp_specification=decomp_specification,
            specification=specification,
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolNeuralLTLSynSolutionSpecPair):
            return False
        else:
            return SpecificationConverterPb2.__eq__(self, __o) and self.solution == __o.solution

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        solution = self.solution.to_csv_fields(**kwargs)
        specification = self.specification_to_csv_fields(**kwargs)

        return {**specification, **solution}

    def to_pb2_NeuralLTLSynSolutionSpecPair(
        self, **kwargs
    ) -> ltl_syn_pb2.NeuralLTLSynSolutionSpecPair:
        return ltl_syn_pb2.NeuralLTLSynSolutionSpecPair(
            **self.specification_to_pb2(**kwargs),
            solution=self.solution.to_pb2_NeuralLTLSynSolution(),
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ToolNeuralLTLSynSolution.csv_field_header(
            **kwargs
        ) + cls.specification_csv_field_header(**kwargs)

    @classmethod
    def from_pb2_NeuralLTLSynSolutionSpecPair(
        cls, pb2_obj: ltl_syn_pb2.NeuralLTLSynSolutionSpecPair, **kwargs
    ) -> "ToolNeuralLTLSynSolutionSpecPair":
        solution = ToolNeuralLTLSynSolution.from_pb2_NeuralLTLSynSolution(pb2_obj.solution)
        decomp_specification, formula_specification = cls.from_specification_pb2(
            pb2_obj.decomp_specification if pb2_obj.HasField("decomp_specification") else None,
            pb2_obj.formula_specification if pb2_obj.HasField("formula_specification") else None,
            **kwargs,
        )

        return cls(
            decomp_specification=decomp_specification,
            formula_specification=formula_specification,
            solution=solution,
        )
