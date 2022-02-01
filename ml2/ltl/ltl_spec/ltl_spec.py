"""Class representing an LTL specification"""

import enum
import json
import logging
import ntpath
import os
from random import sample

SEMANTICS = ["mealy", "moore"]


class LTLSpecSemantics(enum.Enum):
    MEALY = "mealy"
    MOORE = "moore"


class LTLSpec:
    def __init__(
        self,
        inputs: list,
        outputs: list,
        guarantees: list,
        assumptions: list = None,
        name: str = None,
        semantics: str = None,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.guarantees = guarantees
        self.assumptions = assumptions if assumptions else []
        self.name = name
        self.semantics = semantics

        for i, guarantee in enumerate(self.guarantees):
            self.guarantees[i] = guarantee.replace("&&", "&").replace("||", "|")
        for i, assumption in enumerate(self.assumptions):
            self.assumptions[i] = assumption.replace("&&", "&").replace("||", "|")

    @property
    def assumption_str(self):
        parenthesized_assumptions = [f"({assumption})" for assumption in self.assumptions]
        return " & ".join(parenthesized_assumptions)

    @property
    def guarantee_str(self):
        parenthesized_guarantees = [f"({guarantee})" for guarantee in self.guarantees]
        return " & ".join(parenthesized_guarantees)

    @property
    def formula_str(self):
        if self.assumptions:
            return f"({self.assumption_str}) -> ({self.guarantee_str})"
        return self.guarantee_str

    @property
    def input_str(self):
        return ",".join(self.inputs)

    @property
    def output_str(self):
        return ",".join(self.outputs)

    @property
    def num_inputs(self):
        """Number of inputs"""
        return len(self.inputs)

    @property
    def num_outputs(self):
        """Number of outputs"""
        return len(self.outputs)

    @property
    def num_guarantees(self):
        """Number of guarantees"""
        return len(self.guarantees)

    @property
    def num_assumptions(self):
        """Number of assumptions"""
        return len(self.assumptions)

    def rename_aps(self, input_aps, output_aps, random=True, renaming=None):
        """Renames the atomic propositions given new input and output atomic propositions or an explicit renaming"""
        if renaming is None:
            if random:
                renamed_inputs = sample(input_aps, self.num_inputs)
                renamed_outputs = sample(output_aps, self.num_outputs)
            else:
                renamed_inputs = input_aps[: self.num_inputs]
                renamed_outputs = output_aps[: self.num_outputs]
            renaming = dict(zip(self.inputs + self.outputs, renamed_inputs + renamed_outputs))
        else:
            renamed_inputs = map(renaming.get, self.inputs)
            renamed_outputs = map(renaming.get, self.outputs)

        def rename_formula(formula, renaming):
            for (ap, renamed_ap) in renaming.items():
                # TODO doesn't work if ap is substring of other ap
                formula = formula.replace(ap, renamed_ap)
            return formula

        for i, assumption in enumerate(self.assumptions):
            self.assumptions[i] = rename_formula(assumption, renaming)

        for i, guarantee in enumerate(self.guarantees):
            self.guarantees[i] = rename_formula(guarantee, renaming)

        self.inputs = renamed_inputs
        self.outputs = renamed_outputs

        inv_renaming = {v: k for k, v in renaming.items()}
        return inv_renaming

    def to_file(self, file_dir, file_name=None, format="tlsf"):
        if not file_name:
            file_name = self.name
        filepath = os.path.join(file_dir, file_name)
        if format == "tlsf":
            self.to_tlsf_file(filepath)
        elif format == "bosy":
            self.to_bosy_file(filepath)
        else:
            logging.critical("Unkown format %s", format)

    def to_bosy_file(self, filepath):
        # TODO check if format is correct
        assert self.semantics is not None
        with open(filepath, "w") as bosy_file:
            bosy_input = {}
            bosy_input["semantics"] = self.semantics
            bosy_input["inputs"] = self.inputs
            bosy_input["outputs"] = self.outputs
            bosy_input["assumptions"] = self.assumptions
            bosy_input["guarantees"] = self.guarantees
            json.dump(bosy_input, bosy_file, indent=2)

    def to_tlsf_file(self, filepath):
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
            # assumptions
            if self.assumptions:
                tlsf_file.write("  ASSUMPTIONS {\n")
                for a in self.assumptions:
                    a = a.replace("&", "&&")
                    a = a.replace("|", "||")
                    tlsf_file.write(f"    {a};\n")
                tlsf_file.write("  }\n\n")
            # guarantees
            tlsf_file.write("  GUARANTEES {\n")
            for g in self.guarantees:
                g = g.replace("&", "&&")
                g = g.replace("|", "||")
                tlsf_file.write(f"    {g};\n")
            tlsf_file.write("  }\n\n")
            tlsf_file.write("}\n")

    @classmethod
    def from_dict(cls, spec: dict):
        return cls(
            spec["inputs"],
            spec["outputs"],
            spec["guarantees"],
            assumptions=spec["assumptions"] if "assumptions" in spec else None,
            name=spec["name"] if "name" in spec else None,
            semantics=spec["semantics"] if "semantics" in spec else None,
        )

    @classmethod
    def from_bosy_file(cls, filepath):
        with open(filepath, "r") as spec_file:
            spec_dict = json.loads(spec_file.read())
        spec_dict["name"] = ntpath.basename(filepath)
        return cls.from_dict(spec_dict)
