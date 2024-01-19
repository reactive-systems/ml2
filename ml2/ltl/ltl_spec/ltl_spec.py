"""Class representing an LTL specification"""

import enum
import hashlib
import json
import logging
import os
from copy import deepcopy
from random import sample
from typing import Any, Dict, List, Optional

from numpy.random import default_rng

from ...dtypes.binary_ast import BinaryAST
from ...grpc.ltl import ltl_pb2
from ...registry import register_type
from ..ltl_formula import LTLFormula

SEMANTICS = ["mealy", "moore"]


class LTLSpecSemantics(enum.Enum):
    MEALY = "mealy"
    MOORE = "moore"


@register_type
class LTLSpec(LTLFormula):
    def __init__(
        self,
        ast: BinaryAST = None,
        formula: str = None,
        inputs: List[str] = None,
        name: str = None,
        notation: str = None,
        outputs: List[str] = None,
        semantics: str = None,
        tokens: List[str] = None,
    ):
        self.inputs = inputs if inputs else []
        self.outputs = outputs if outputs else []
        self.name = name
        self.semantics = semantics

        super().__init__(
            ast=ast,
            formula=formula.replace("&&", "&").replace("||", "|") if formula is not None else None,
            notation=notation,
            tokens=tokens,
        )

    @property
    def unique_mod_aps(self) -> int:
        """Uniqueness modulo atomic propositions"""
        s = deepcopy(self)
        s.reset_aps()
        s.rename_aps(random=False)
        s.semantics = None
        return s.cr_hash

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.to_str("prefix"),
                        self.semantics,
                        sorted([i for i in self.inputs]),
                        sorted([o for o in self.outputs]),
                    )
                ).encode()
            ).hexdigest(),
            16,
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        else:
            return self.cr_hash == __o.cr_hash

    @property
    def input_str(self):
        return ",".join(self.inputs)

    @property
    def output_str(self):
        return ",".join(self.outputs)

    @property
    def num_inputs(self) -> int:
        """Number of inputs"""
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        """Number of outputs"""
        return len(self.outputs)

    def deduce_inputs(self, inputs: Optional[List[str]] = None) -> List[str]:
        if inputs is None:
            assert self.inputs is not None
            inputs = self.inputs
        return sorted(list(filter(lambda i: i in self.ast.leaves, inputs)))

    def deduce_outputs(self, outputs: Optional[List[str]] = None) -> List[str]:
        if outputs is None:
            assert self.outputs is not None
            outputs = self.outputs
        return sorted(list(filter(lambda i: i in self.ast.leaves, outputs)))

    def reset_inputs(self, inputs: Optional[List[str]] = None) -> None:
        self.inputs = self.deduce_inputs(inputs)

    def reset_outputs(self, outputs: Optional[List[str]] = None) -> None:
        self.outputs = self.deduce_outputs(outputs)

    def reset_aps(
        self, inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None
    ) -> None:
        self.reset_inputs(inputs)
        self.reset_outputs(outputs)

    def to_pb2_LTLSpecification(self, **kwargs):
        return ltl_pb2.LTLSpecification(
            inputs=self.inputs, outputs=self.outputs, formula=self.to_pb2_LTLFormula(**kwargs)
        )

    @classmethod
    def from_pb2_LTLSpecification(cls, pb2_LTLSpecification, **kwargs) -> "LTLSpec":
        inputs = [str(i) for i in pb2_LTLSpecification.inputs]
        outputs = [str(o) for o in pb2_LTLSpecification.outputs]

        return cls.from_dict(
            {
                "formula": pb2_LTLSpecification.formula.formula,
                "notation": pb2_LTLSpecification.formula.notation,
                "inputs": inputs,
                "outputs": outputs,
            },
            **kwargs,
        )

    def rename_aps(
        self,
        input_aps: List[str] = None,
        output_aps: List[str] = None,
        random: bool = True,
        random_weighted: Optional[Dict[str, float]] = None,
        renaming: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Renames the atomic propositions given new input and output atomic propositions or an explicit renaming"""

        def weighted_random(aps: List[str], random_weighted: Dict[str, float], number: int):
            probs = []
            for a in aps:
                probs.append(random_weighted[a] if a in random_weighted.keys() else 1)
            frac = 1 / sum(probs)
            probs = [p * frac for p in probs]

            rng = default_rng()
            return list(rng.choice(aps, number, replace=False, p=probs))

        if renaming is None:
            if input_aps is None:
                input_aps = ["i" + str(i) for i in range(self.num_inputs)]
            if output_aps is None:
                output_aps = ["o" + str(i) for i in range(self.num_outputs)]
            if random:
                if random_weighted is None:
                    renamed_inputs = sample(input_aps, self.num_inputs)
                    renamed_outputs = sample(output_aps, self.num_outputs)
                else:
                    renamed_inputs = (
                        weighted_random(
                            input_aps, random_weighted=random_weighted, number=self.num_inputs
                        )
                        if self.num_inputs != 0
                        else []
                    )
                    renamed_outputs = (
                        weighted_random(
                            output_aps, random_weighted=random_weighted, number=self.num_outputs
                        )
                        if self.num_outputs != 0
                        else []
                    )
            else:
                renamed_inputs = input_aps[: self.num_inputs]
                renamed_outputs = output_aps[: self.num_outputs]
            renaming = dict(zip(self.inputs + self.outputs, renamed_inputs + renamed_outputs))
        else:
            renamed_inputs = map(renaming.get, self.inputs)
            renamed_outputs = map(renaming.get, self.outputs)

        self.rename(renaming)

        inv_renaming = {v: k for k, v in renaming.items()}
        return inv_renaming

    def rename(self, rename: Dict[str, str]):
        self.inputs = [rename[input] for input in self.inputs] if self.inputs is not None else None
        self.outputs = (
            [rename[output] for output in self.outputs] if self.outputs is not None else None
        )
        super().rename(rename)

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        d = self.to_dict(notation=notation, **kwargs)
        d["inputs"] = self.input_str
        d["outputs"] = self.output_str
        return d

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return [
            "inputs",
            "outputs",
            "formula",
            "name",
            "notation",
            "semantics",
            "inputs",
            "outputs",
        ]

    def to_dict(self, notation: str = None, **kwargs) -> dict:
        d = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "formula": self.to_str(notation=notation),
        }
        if self.name is not None:
            d["name"] = self.name
        if notation is not None:
            d["notation"] = notation.value
        if self.semantics is not None:
            d["semantics"] = self.semantics
        return d

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, Any], **kwargs) -> "LTLSpec":
        d = fields.copy()
        d["inputs"] = d["inputs"].split(",") if d["inputs"] != "" else []
        d["outputs"] = d["outputs"].split(",") if d["outputs"] != "" else []
        return cls.from_dict(d=d)

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> "LTLSpec":
        return cls(
            formula=d["formula"],
            inputs=d["inputs"],
            name=d.get("name", None),
            notation=d.get("notation", "infix"),
            outputs=d["outputs"],
            semantics=d.get("semantics", None),
        )

    def to_file(
        self, file_dir: str, filename: Optional[str] = None, file_format: str = "tlsf"
    ) -> None:
        if filename is None:
            if self.name is not None:
                filename = self.name
            else:
                raise ValueError("Filename not specified nor name of LTL specification set")
            if file_format == "tlsf":
                filename += ".tlsf"
            elif file_format == "bosy":
                filename += ".json"

        filepath = os.path.join(file_dir, filename)

        if file_format == "tlsf":
            self.to_tlsf_file(filepath)
        elif file_format == "bosy":
            self.to_bosy_file(filepath)
        else:
            logging.critical("Unknown format %s", file_format)

    def to_bosy_file(self, filepath: str) -> None:
        assert self.semantics is not None
        with open(filepath, "w") as bosy_file:
            bosy_input = {}
            bosy_input["semantics"] = self.semantics
            bosy_input["inputs"] = self.inputs
            bosy_input["outputs"] = self.outputs
            bosy_input["assumptions"] = []
            bosy_input["guarantees"] = [self.to_str()]
            json.dump(bosy_input, bosy_file, indent=2)

    def to_tlsf_file(self, filepath: str) -> None:
        with open(filepath, "w") as tlsf_file:
            # info section
            tlsf_file.write("INFO {\n")
            tlsf_file.write('  TITLE:       "title"\n')
            tlsf_file.write('  DESCRIPTION: "description"\n')
            tlsf_file.write("  SEMANTICS:   Mealy\n")
            tlsf_file.write("  TARGET:      Mealy\n")
            tlsf_file.write("}\n\n")
            # main section
            tlsf_file.write("MAIN {\n")
            # inputs
            tlsf_file.write("  INPUTS {\n")
            for i in self.inputs:
                tlsf_file.write(f"    {i};\n")
            tlsf_file.write("  }\n\n")
            # outputs
            tlsf_file.write("  OUTPUTS {\n")
            for o in self.outputs:
                tlsf_file.write(f"    {o};\n")
            tlsf_file.write("  }\n\n")
            # guarantees
            tlsf_file.write("  GUARANTEES {\n")
            f_str = self.to_str(notation="infix")
            f_str = f_str.replace("&", "&&")
            f_str = f_str.replace("|", "||")
            tlsf_file.write(f"    {f_str};\n")
            tlsf_file.write("  }\n\n")
            tlsf_file.write("}\n")
