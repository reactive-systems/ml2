"""Protobuf converter"""

import logging
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple, Union

from google.protobuf import duration_pb2

from ...aiger import AIGERCircuit
from ...grpc.aiger import aiger_pb2
from ...grpc.ltl import ltl_pb2
from ...grpc.mealy import mealy_pb2
from ...ltl.ltl_spec import DecompLTLSpec, LTLSpec
from ...mealy import MealyMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecificationConverterPb2:
    def __init__(
        self,
        formula_specification: Optional[LTLSpec] = None,
        decomp_specification: Optional[DecompLTLSpec] = None,
        specification: Optional[Union[LTLSpec, DecompLTLSpec]] = None,
    ) -> None:
        if specification is not None and isinstance(specification, DecompLTLSpec):
            decomp_specification = specification
        elif specification is not None and isinstance(specification, LTLSpec):
            formula_specification = specification
        if formula_specification is not None and decomp_specification is not None:
            raise ValueError("Only one of circuit or mealy_machine can be set.")
        self.formula_specification = formula_specification
        self.decomp_specification = decomp_specification

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SpecificationConverterPb2):
            return False
        else:
            return (
                self.decomp_specification == __o.decomp_specification
                and self.formula_specification == __o.formula_specification
            )

    @property
    def which_of_specification(self) -> Literal["formula_specification", "decomp_specification"]:
        if self.formula_specification is not None and self.decomp_specification is None:
            return "formula_specification"
        elif self.decomp_specification is not None and self.formula_specification is None:
            return "decomp_specification"
        else:
            raise ValueError(
                "Exactly one of formula_specification or decomp_specification should be set."
            )

    @property
    def specification(self) -> Union[LTLSpec, DecompLTLSpec]:
        if self.formula_specification is not None and self.decomp_specification is None:
            return self.formula_specification
        elif self.decomp_specification is not None and self.formula_specification is None:
            return self.decomp_specification
        else:
            raise ValueError(
                "Exactly one of formula_specification or decomp_specification should be set."
            )

    @property
    def specification_dict(self) -> Dict[str, Union[LTLSpec, DecompLTLSpec]]:
        return {self.which_of_specification: self.specification}

    def specification_to_pb2(
        self, **kwargs
    ) -> Dict[str, Union[ltl_pb2.LTLSpecification, ltl_pb2.DecompLTLSpecification]]:
        if self.which_of_specification == "formula_specification":
            assert self.formula_specification is not None
            return {
                self.which_of_specification: self.specification.to_pb2_LTLSpecification(**kwargs)
            }
        elif self.which_of_specification == "decomp_specification":
            assert self.decomp_specification is not None
            return {
                self.which_of_specification: self.decomp_specification.to_pb2_DecompLTLSpecification(
                    **kwargs
                )
            }
        else:
            raise ValueError(
                "Exactly one of formula_specification or decomp_specification should be set."
            )

    @staticmethod
    def from_specification_pb2(
        decomp_specification_pb2: Optional[ltl_pb2.DecompLTLSpecification],
        formula_specification_pb2: Optional[ltl_pb2.LTLSpecification],
        **kwargs
    ) -> Tuple[Optional[DecompLTLSpec], Optional[LTLSpec]]:
        decomp_specification = (
            DecompLTLSpec.from_pb2_DecompLTLSpecification(decomp_specification_pb2, **kwargs)
            if decomp_specification_pb2 is not None
            else None
        )
        formula_specification = (
            LTLSpec.from_pb2_LTLSpecification(formula_specification_pb2, **kwargs)
            if formula_specification_pb2 is not None
            else None
        )
        return decomp_specification, formula_specification

    def specification_to_csv_fields(self, **kwargs) -> Dict[str, str]:
        if self.which_of_specification == "formula_specification":
            assert self.formula_specification is not None
            return self.formula_specification.to_csv_fields(**kwargs)
        elif self.which_of_specification == "decomp_specification":
            assert self.decomp_specification is not None
            return self.decomp_specification.to_csv_fields(**kwargs)
        else:
            raise ValueError(
                "Exactly one of formula_specification or decomp_specification should be set."
            )

    @classmethod
    def specification_csv_field_header(cls, **kwargs) -> List[str]:
        return LTLSpec.csv_field_header(**kwargs) + DecompLTLSpec.csv_field_header(**kwargs)


class SystemConverterPb2:
    def __init__(
        self,
        circuit: Optional[AIGERCircuit] = None,
        mealy_machine: Optional[MealyMachine] = None,
        realizable: Optional[bool] = None,
    ) -> None:
        if circuit is not None and mealy_machine is not None:
            raise ValueError("Only one of circuit or mealy_machine can be set.")
        self.circuit = circuit
        self.mealy_machine = mealy_machine
        self._realizable = realizable

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SystemConverterPb2):
            return False
        else:
            return self.realizable == __o.realizable and self.system == __o.system

    @property
    def realizable(self) -> Optional[bool]:
        return self._realizable

    @property
    def which_of_system(self) -> Optional[Literal["circuit", "mealy_machine"]]:
        if self.circuit is not None and self.mealy_machine is None:
            return "circuit"
        elif self.mealy_machine is not None and self.circuit is None:
            return "mealy_machine"
        elif self.circuit is None and self.mealy_machine is None:
            return None
        else:
            raise ValueError("Only one of circuit or mealy_machine should be set.")

    @property
    def system(self) -> Optional[Union[AIGERCircuit, MealyMachine]]:
        if self.circuit is not None and self.mealy_machine is None:
            return self.circuit
        elif self.mealy_machine is not None and self.circuit is None:
            return self.mealy_machine
        elif self.circuit is None and self.mealy_machine is None:
            return None
        else:
            raise ValueError("Only one of circuit or mealy_machine should be set.")

    @property
    def system_dict(self) -> Dict[str, Union[AIGERCircuit, MealyMachine]]:
        return (
            {self.which_of_system: self.system}
            if (self.which_of_system is not None and self.system is not None)
            else {}
        )

    @staticmethod
    def from_system_pb2(
        circuit_pb2: Optional[aiger_pb2.AigerCircuit],
        mealy_machine_pb2: Optional[mealy_pb2.MealyMachine],
        **kwargs
    ) -> Tuple[Optional[AIGERCircuit], Optional[MealyMachine]]:
        circuit = (
            AIGERCircuit.from_str(circuit_pb2.circuit, **kwargs)
            if circuit_pb2 is not None
            else None
        )
        mealy_machine = (
            MealyMachine.from_hoa(mealy_machine_pb2.machine)
            if mealy_machine_pb2 is not None
            else None
        )
        return circuit, mealy_machine

    def system_to_pb2(
        self, **kwargs
    ) -> Dict[str, Union[aiger_pb2.AigerCircuit, mealy_pb2.MealyMachine]]:
        if self.which_of_system == "circuit":
            assert self.circuit is not None
            return {self.which_of_system: aiger_pb2.AigerCircuit(circuit=self.circuit.to_str())}
        elif self.which_of_system == "mealy_machine":
            assert self.mealy_machine is not None
            assert self.realizable is not None
            return {
                self.which_of_system: mealy_pb2.MealyMachine(
                    machine=self.mealy_machine.to_hoa(realizable=self.realizable, **kwargs)
                )
            }
        elif self.which_of_system is None:
            return {}
        else:
            raise ValueError("Only one of circuit or mealy_machine should be set.")

    def system_to_csv_fields(self, **kwargs) -> Dict[str, str]:
        if self.which_of_system == "circuit":
            assert self.circuit is not None
            return self.circuit.to_csv_fields(**kwargs)
        elif self.which_of_system == "mealy_machine":
            assert self.mealy_machine is not None
            assert self.realizable is not None
            return self.mealy_machine.to_csv_fields(**kwargs)
        elif self.which_of_system is None:
            return {}
        else:
            raise ValueError("Only one of circuit or mealy_machine should be set.")

    @classmethod
    def system_csv_field_header(cls, **kwargs) -> List[str]:
        return AIGERCircuit.csv_field_header(**kwargs) + MealyMachine.csv_field_header(**kwargs)


class TimeConverterPb2:
    def __init__(
        self,
        time: Optional[timedelta] = None,
    ) -> None:
        self.time = time if time is not None else timedelta(0)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, TimeConverterPb2):
            return False
        else:
            return self.time == __o.time

    @property
    def time_seconds(self) -> float:
        return self.time.total_seconds()

    def time_to_pb2(self, **kwargs) -> duration_pb2.Duration:
        time = duration_pb2.Duration()
        time.FromTimedelta(self.time)
        return time

    @staticmethod
    def from_time_tb2(pb2_obj: duration_pb2.Duration) -> timedelta:
        return pb2_obj.ToTimedelta()

    def time_to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"duration": str(self.time_seconds)}

    @classmethod
    def time_csv_field_header(cls, **kwargs) -> List[str]:
        return ["duration"]
