"""Tool LTL model checking problem"""

import json
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from ...aiger import AIGERCircuit
from ...dtypes import CatSeq, CSVLoggable
from ...grpc.ltl import ltl_mc_pb2
from ...grpc.trace import trace_pb2
from ...ltl.ltl_mc import LTLMCSolution, LTLMCStatus
from ...ltl.ltl_spec import DecompLTLSpec, LTLSpec
from ...ltl.ltl_syn import LTLRealStatus
from ...mealy import MealyMachine
from ...trace import Trace
from .pb2_converter import SpecificationConverterPb2, SystemConverterPb2, TimeConverterPb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolLTLMCProblem(CSVLoggable, SpecificationConverterPb2, SystemConverterPb2):
    def __init__(
        self,
        realizable: bool,
        parameters: Optional[Dict[str, Any]] = None,
        formula_specification: Optional[LTLSpec] = None,
        decomp_specification: Optional[DecompLTLSpec] = None,
        specification: Optional[Union[LTLSpec, DecompLTLSpec]] = None,
        circuit: Optional[AIGERCircuit] = None,
        mealy_machine: Optional[MealyMachine] = None,
    ) -> None:
        self.parameters = parameters if parameters is not None else {}
        SpecificationConverterPb2.__init__(
            self, formula_specification, decomp_specification, specification
        )
        SystemConverterPb2.__init__(self, circuit, mealy_machine, realizable)
        if self.formula_specification is None and self.decomp_specification is None:
            raise ValueError(
                "At least one of formula_specification or decomp_specification needs to be set."
            )
        if self.circuit is None and self.mealy_machine is None:
            raise ValueError("At least one of circuit or mealy_machine needs to be set.")

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolLTLMCProblem):
            return False
        else:
            return (
                self.parameters == __o.parameters
                and SpecificationConverterPb2.__eq__(self, __o)
                and SystemConverterPb2.__eq__(self, __o)
            )

    @property
    def realizable(self) -> bool:
        assert self._realizable is not None
        return self._realizable

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        assert self.realizable is not None

        params = {"parameters": json.dumps(self.parameters)}
        system = self.system_to_csv_fields(**kwargs)
        realizable = {"realizable": str(self.realizable)}
        spec = self.specification_to_csv_fields(**kwargs)

        return {**params, **system, **realizable, **spec}

    def to_pb2_LTLMCProblem(self, **kwargs) -> ltl_mc_pb2.LTLMCProblem:
        spec = self.specification_to_pb2(**kwargs)
        system = self.system_to_pb2(**kwargs)
        assert self.realizable is not None

        params = {k: json.dumps(v) for k, v in self.parameters.items()}
        return ltl_mc_pb2.LTLMCProblem(
            realizable=self.realizable,
            parameters=params,
            **system,
            **spec,
        )

    @classmethod
    def from_aiger_verification_pair(
        cls,
        formula: LTLSpec,
        solution: CatSeq[LTLRealStatus, AIGERCircuit],
        parameters: Dict[str, Any],
    ):
        return cls(
            parameters=parameters,
            realizable=solution.cat.realizable,
            specification=formula,
            circuit=solution.seq,
        )

    @classmethod
    def from_pb2_LTLMCProblem(
        cls, pb2_obj: ltl_mc_pb2.LTLMCProblem, **kwargs
    ) -> "ToolLTLMCProblem":
        decomp_specification, formula_specification = cls.from_specification_pb2(
            pb2_obj.decomp_specification if pb2_obj.HasField("decomp_specification") else None,
            pb2_obj.formula_specification if pb2_obj.HasField("formula_specification") else None,
            **kwargs,
        )

        parameters = {k: json.loads(v) for k, v in pb2_obj.parameters.items()}

        circuit, mealy_machine = cls.from_system_pb2(
            pb2_obj.circuit if pb2_obj.HasField("circuit") else None,
            pb2_obj.mealy_machine if pb2_obj.HasField("mealy_machine") else None,
            **kwargs,
        )

        return cls(
            parameters=parameters,
            decomp_specification=decomp_specification,
            formula_specification=formula_specification,
            realizable=pb2_obj.realizable,
            circuit=circuit,
            mealy_machine=mealy_machine,
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return (
            ["parameters", "realizable"]
            + cls.specification_csv_field_header(**kwargs)
            + cls.system_csv_field_header(**kwargs)
        )


class ToolLTLMCSolution(CSVLoggable, TimeConverterPb2):
    def __init__(
        self,
        status: LTLMCStatus,
        detailed_status: str,
        tool: str,
        time: Optional[timedelta] = None,
        counterexample: Optional[Trace] = None,
    ) -> None:
        self.status = status
        self.detailed_status = detailed_status
        self.tool = tool
        self.counterexample = counterexample
        TimeConverterPb2.__init__(self, time)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ToolLTLMCSolution):
            return False
        else:
            return (
                self.detailed_status == __o.detailed_status
                and self.status == __o.status
                and self.tool == __o.tool
                and TimeConverterPb2.__eq__(self, __o)
                and self.counterexample == __o.counterexample
            )

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        status = self.status.to_csv_fields()
        detailed_status = (
            {"detailed_status": self.detailed_status} if self.detailed_status != "" else {}
        )
        tool = {"tool": self.tool}
        time = self.time_to_csv_fields(**kwargs)
        counterexample = (
            self.counterexample.to_csv_fields(**kwargs) if self.counterexample is not None else {}
        )

        return {**status, **detailed_status, **tool, **time, **counterexample}

    def to_LTLMCSolution(self):
        return LTLMCSolution(
            status=self.status,
            trace=self.counterexample,
            time=self.time_seconds,
            tool=self.tool,
        )

    def to_pb2_LTLMCSolution(self, **kwargs) -> ltl_mc_pb2.LTLMCSolution:
        status, detailed_status = self.status.to_pb2_LTLMCStatus(**kwargs)
        detailed_status = self.detailed_status if self.detailed_status != "" else detailed_status
        time = self.time_to_pb2(**kwargs)
        counterexample = (
            self.counterexample.to_str(**kwargs) if self.counterexample is not None else ""
        )

        return ltl_mc_pb2.LTLMCSolution(
            status=status,
            detailed_status=detailed_status,
            tool=self.tool,
            time=time,
            counterexample=trace_pb2.Trace(trace=counterexample),
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return (
            ["detailed_status"]
            + LTLMCStatus.csv_field_header(**kwargs)
            + Trace.csv_field_header(**kwargs)
            + ["tool"]
            + cls.time_csv_field_header(**kwargs)
        )

    @classmethod
    def from_pb2_LTLMCSolution(
        cls, pb2_obj: ltl_mc_pb2.LTLMCSolution, **kwargs
    ) -> "ToolLTLMCSolution":
        status: LTLMCStatus = LTLMCStatus.from_pb2_LTLMCStatus(
            pb2_obj.status, pb2_obj.detailed_status, **kwargs
        )
        detailed_status: str = pb2_obj.detailed_status
        tool: str = pb2_obj.tool
        time: timedelta = cls.from_time_tb2(pb2_obj.time, **kwargs)
        try:
            trace = Trace.from_str(pb2_obj.counterexample.trace)
        except Exception:
            trace = None

        return cls(
            status=status,
            detailed_status=detailed_status,
            tool=tool,
            time=time,
            counterexample=trace,
        )
