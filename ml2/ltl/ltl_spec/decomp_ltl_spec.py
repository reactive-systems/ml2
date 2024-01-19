"""Class representing a decomposed LTL specification"""

import hashlib
import json
import ntpath
from copy import deepcopy
from random import sample
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from numpy.random import default_rng

from ...dtypes import BinaryAST, DecompBinaryExpr, DecompBinaryExprPair
from ...grpc.ltl import ltl_pb2
from ...registry import register_type
from ...utils.list_utils import join_lists
from .ltl_spec import LTLSpec

T = TypeVar("T", bound="LTLProperties")


class LTLProperties(DecompBinaryExpr, LTLSpec):
    BINARY_EXPR_TYPE = LTLSpec

    def __init__(
        self,
        sub_exprs: Optional[List[LTLSpec]] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        name: Optional[str] = None,
        semantics: Optional[str] = None,
    ):
        self.inputs = inputs if inputs else []
        self.outputs = outputs if outputs else []
        self.name = name
        self.semantics = semantics

        DecompBinaryExpr.__init__(self, sub_exprs=sub_exprs)

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        sorted([sub.cr_hash for sub in self.sub_exprs]),
                        self.semantics,
                        sorted([i for i in self.inputs]),
                        sorted([o for o in self.outputs]),
                    )
                ).encode()
            ).hexdigest(),
            16,
        )

    def rename(self, rename: Dict[str, str]):
        self.inputs = [rename[input] for input in self.inputs]
        self.outputs = [rename[output] for output in self.outputs]
        super().rename(rename)

    def condense(self) -> LTLSpec:
        return LTLSpec(
            formula=self.to_str(), inputs=self.deduce_inputs(), outputs=self.deduce_ouputs()
        )

    def deduce_inputs(self, inputs: Optional[List[str]] = None) -> List[str]:
        return sorted(
            list(
                set(
                    {
                        el
                        for sublist in (g.deduce_inputs(inputs) for g in self.sub_exprs)
                        for el in sublist
                    }
                )
            )
        )

    def deduce_ouputs(self, outputs: Optional[List[str]] = None) -> List[str]:
        return sorted(
            list(
                set(
                    {
                        el
                        for sublist in (g.deduce_outputs(outputs) for g in self.sub_exprs)
                        for el in sublist
                    }
                )
            )
        )

    def reset_inputs(self, inputs: Optional[List[str]] = None):
        (g.reset_inputs(inputs) for g in self.sub_exprs)
        self.inputs = self.deduce_inputs()

    def reset_outputs(self, outputs: Optional[List[str]] = None):
        (g.reset_outputs(outputs) for g in self.sub_exprs)
        self.outputs = self.deduce_ouputs()

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {
            "inputs": self.input_str,
            "outputs": self.output_str,
            "properties": ",".join([a.to_str(notation=notation) for a in self.sub_exprs]),
        }
        if self.name is not None:
            fields["name"] = self.name
        if notation is not None:
            fields["notation"] = notation.value
        if self.semantics is not None:
            fields["semantics"] = self.semantics
        return fields

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["inputs", "outputs", "properties", "name", "notation", "semantics"]

    @classmethod
    def _from_csv_fields(cls: Type[T], fields: Dict[str, str], **kwargs) -> T:
        d = fields.copy()
        d["inputs"] = d["inputs"].split(",") if d["inputs"] != "" else []
        d["outputs"] = d["outputs"].split(",") if d["outputs"] != "" else []
        if "properties" in d and d["properties"] != "":
            d["properties"] = d["properties"].split(",")
        else:
            d["properties"] = None
        return cls.from_dict(d=d, **kwargs)

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any], **kwargs) -> T:
        return cls(
            sub_exprs=[
                LTLSpec.from_dict(
                    d={
                        "formula": a,
                        "inputs": d["inputs"],
                        "outputs": d["outputs"],
                        "notation": d.get("notation", "infix"),
                        "semantics": d.get("semantics", None),
                    },
                )
                for a in d["properties"]
            ]
            if "properties" in d and d["properties"] is not None
            else None,
            inputs=d["inputs"],
            outputs=d["outputs"],
            name=d.get("name", None),
            semantics=d.get("semantics", None),
        )

    @staticmethod
    def comp_asts(asts: List[BinaryAST]) -> BinaryAST:
        if len(asts) == 0:
            return None
        comp_ast = asts[0]
        for ast in asts[1:]:
            comp_ast = BinaryAST("&", comp_ast, ast)
        return comp_ast

    @staticmethod
    def comp_strs(strs: List[str], notation: str = None) -> str:
        if notation == "infix":
            return " & ".join([f"( {a} )" for a in strs])
        elif notation == "infix-no-pars":
            return " & ".join(strs)
        elif notation == "prefix":
            return "& " * max(len(strs) - 1, 0) + " ".join(strs)
        else:
            raise ValueError(f"Unknown notation {notation}")

    @staticmethod
    def comp_token_lists(token_lists: List[List[str]], notation: str = None) -> List[str]:
        if notation == "infix":
            return join_lists("&", [["("] + l + [")"] for l in token_lists])
        elif notation == "infix-no-pars":
            return join_lists("&", token_lists)
        elif notation == "prefix":
            return ["&"] * max(len(token_lists) - 1, 0) + [t for l in token_lists for t in l]
        else:
            raise ValueError(f"Unknown notation {notation}")


@register_type
class LTLAssumptions(LTLProperties):
    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {
            "inputs": self.input_str,
            "outputs": self.output_str,
            "assumptions": ",".join([a.to_str(notation=notation) for a in self.sub_exprs]),
        }
        if self.name is not None:
            fields["name"] = self.name
        if notation is not None:
            fields["notation"] = notation.value
        if self.semantics is not None:
            fields["semantics"] = self.semantics
        return fields

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["inputs", "outputs", "assumptions", "name", "notation", "semantics"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLAssumptions":
        d = fields.copy()
        d["inputs"] = d["inputs"].split(",") if d["inputs"] != "" else []
        d["outputs"] = d["outputs"].split(",") if d["outputs"] != "" else []
        if "assumptions" in d and d["assumptions"] != "":
            d["assumptions"] = d["assumptions"].split(",")
        else:
            d["assumptions"] = None
        return cls.from_dict(d=d, **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs) -> "LTLAssumptions":
        return cls(
            sub_exprs=[
                LTLSpec.from_dict(
                    d={
                        "formula": a,
                        "inputs": d["inputs"],
                        "outputs": d["outputs"],
                        "notation": d.get("notation", "infix"),
                        "semantics": d.get("semantics", None),
                    },
                )
                for a in d["assumptions"]
            ]
            if "assumptions" in d and d["assumptions"] is not None
            else None,
            inputs=d["inputs"],
            outputs=d["outputs"],
            name=d.get("name", None),
            semantics=d.get("semantics", None),
        )


@register_type
class LTLGuarantees(LTLProperties):
    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {
            "inputs": self.input_str,
            "outputs": self.output_str,
            "guarantees": ",".join([g.to_str(notation=notation) for g in self.sub_exprs]),
        }
        if self.name is not None:
            fields["name"] = self.name
        if notation is not None:
            fields["notation"] = notation.value
        if self.semantics is not None:
            fields["semantics"] = self.semantics
        return fields

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["inputs", "outputs", "guarantees", "name", "notation", "semantics"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "LTLGuarantees":
        d = fields.copy()
        d["inputs"] = d["inputs"].split(",") if d["inputs"] != "" else []
        d["outputs"] = d["outputs"].split(",") if d["outputs"] != "" else []
        if "guarantees" in d and d["guarantees"] != "":
            d["guarantees"] = d["guarantees"].split(",")
        else:
            d["guarantees"] = None
        return cls.from_dict(d=d, **kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs) -> "LTLGuarantees":
        return cls(
            sub_exprs=[
                LTLSpec.from_dict(
                    d={
                        "formula": g,
                        "inputs": d["inputs"],
                        "outputs": d["outputs"],
                        "notation": d.get("notation", "infix"),
                        "semantics": d.get("semantics", None),
                    },
                )
                for g in d["guarantees"]
            ]
            if "guarantees" in d and d["guarantees"] is not None
            else None,
            inputs=d["inputs"],
            outputs=d["outputs"],
            name=d.get("name", None),
            semantics=d.get("semantics", None),
        )


@register_type
class DecompLTLSpec(DecompBinaryExprPair, LTLSpec):
    def __init__(
        self,
        assumptions: LTLAssumptions,
        guarantees: LTLGuarantees,
        inputs: Optional[List[str]],
        outputs: Optional[List[str]],
        name: Optional[str] = None,
        semantics: Optional[str] = None,
    ):
        self.inputs: List[str] = inputs if inputs is not None else []
        self.outputs: List[str] = outputs if outputs is not None else []
        self.name = name
        self.semantics = semantics

        DecompBinaryExprPair.__init__(self, fst=assumptions, snd=guarantees)

    @property
    def unique_mod_aps(self) -> int:
        """Uniqueness modulo atomic propositions and property order"""
        guarantees = [g.unique_mod_aps for g in self.guarantees]
        assumptions = [a.unique_mod_aps for a in self.assumptions]
        return int(
            hashlib.sha3_224(str((sorted(assumptions), sorted(guarantees))).encode()).hexdigest(),
            16,
        )

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.assumptions.cr_hash,
                        self.guarantees.cr_hash,
                        self.semantics,
                        sorted([i for i in self.inputs]),
                        sorted([o for o in self.outputs]),
                    )
                ).encode()
            ).hexdigest(),
            16,
        )

    @property
    def assumptions(self) -> LTLAssumptions:
        return self[0]  # type: ignore

    @property
    def guarantees(self) -> LTLGuarantees:
        return self[1]  # type: ignore

    def formula_str(self, notation: Optional[str] = None) -> str:
        if self.assumptions:
            return f"({self.assumption_str(notation=notation)}) -> ({self.guarantee_str(notation=notation)})"
        return self.guarantee_str(notation=notation)

    def assumption_str(self, notation: Optional[str] = None) -> str:
        return self.assumptions.to_str(notation=notation)

    def guarantee_str(self, notation: Optional[str] = None) -> str:
        return self.guarantees.to_str(notation=notation)

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
                input_aps.sort()
            if output_aps is None:
                output_aps = ["o" + str(i) for i in range(self.num_outputs)]
                output_aps.sort()
            if random:
                if random_weighted is None:
                    renamed_inputs = sample(input_aps, self.num_inputs)
                    renamed_outputs = sample(output_aps, self.num_outputs)
                else:
                    renamed_inputs = weighted_random(
                        input_aps, random_weighted=random_weighted, number=self.num_inputs
                    )
                    renamed_outputs = weighted_random(
                        output_aps, random_weighted=random_weighted, number=self.num_outputs
                    )
            else:
                renamed_inputs = input_aps[: self.num_inputs]
                renamed_outputs = output_aps[: self.num_outputs]
            renaming = dict(zip(self.inputs + self.outputs, renamed_inputs + renamed_outputs))
        else:
            renamed_inputs = list(map(renaming.get, self.inputs))
            renamed_outputs = list(map(renaming.get, self.outputs))

        self.rename(rename=renaming)
        self.assumptions.rename(rename=renaming)
        self.guarantees.rename(rename=renaming)

        inv_renaming = {v: k for k, v in renaming.items()}
        return inv_renaming

    def _to_csv_fields(self, notation: str = None, **kwargs) -> Dict[str, str]:
        fields = {
            "inputs": ",".join(self.inputs),
            "outputs": ",".join(self.outputs),
            "assumptions": self.assumptions.to_csv_fields(notation=notation)["assumptions"],
            "guarantees": self.guarantees.to_csv_fields(notation=notation)["guarantees"],
        }
        if self.name is not None:
            fields["name"] = self.name
        if notation is not None:
            fields["notation"] = notation.value
        if self.semantics is not None:
            fields["semantics"] = self.semantics
        return fields

    def to_bosy_file(self, filepath: str) -> None:
        assert self.semantics is not None
        with open(filepath, "w") as bosy_file:
            bosy_input = {}
            bosy_input["semantics"] = self.semantics
            bosy_input["inputs"] = self.inputs
            bosy_input["outputs"] = self.outputs
            bosy_input["assumptions"] = [a.to_str() for a in self.assumptions]
            bosy_input["guarantees"] = [g.to_str() for g in self.guarantees]
            json.dump(bosy_input, bosy_file, indent=2)

    def to_pb2_DecompLTLSpecification(self, **kwargs):
        return ltl_pb2.DecompLTLSpecification(
            inputs=self.inputs,
            outputs=self.outputs,
            guarantees=[g.to_pb2_LTLFormula(**kwargs) for g in self.guarantees],
            assumptions=[a.to_pb2_LTLFormula(**kwargs) for a in self.assumptions],
        )

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
            # assumptions
            if self.assumptions:
                tlsf_file.write("  ASSUMPTIONS {\n")
                for a in self.assumptions:
                    a_str = a.to_str(notation="infix")
                    a_str = a_str.replace("&", "&&")
                    a_str = a_str.replace("|", "||")
                    tlsf_file.write(f"    {a_str};\n")
                tlsf_file.write("  }\n\n")
            # guarantees
            tlsf_file.write("  GUARANTEES {\n")
            for g in self.guarantees:
                g_str = g.to_str(notation="infix")
                g_str = g_str.replace("&", "&&")
                g_str = g_str.replace("|", "||")
                tlsf_file.write(f"    {g_str};\n")
            tlsf_file.write("  }\n\n")
            tlsf_file.write("}\n")

    def condense_guaranties(
        self, max_ast_size: Union[int, Literal["inf"]] = 50
    ) -> "DecompLTLSpec":
        condensed = self._condense_properties(
            properties=self.guarantees, max_ast_size=max_ast_size
        )
        g = LTLGuarantees(sub_exprs=condensed)
        g.inputs = g.deduce_inputs()
        g.outputs = g.deduce_ouputs()
        return DecompLTLSpec(
            assumptions=self.assumptions, guarantees=g, inputs=self.inputs, outputs=self.outputs
        )

    def condense_assumptions(
        self, max_ast_size: Union[int, Literal["inf"]] = 50
    ) -> "DecompLTLSpec":
        condensed = self._condense_properties(
            properties=self.assumptions, max_ast_size=max_ast_size
        )
        a = LTLAssumptions(sub_exprs=condensed)
        a.inputs = a.deduce_inputs()
        a.outputs = a.deduce_ouputs()
        return DecompLTLSpec(
            assumptions=a, guarantees=self.guarantees, inputs=self.inputs, outputs=self.outputs
        )

    def condense_properties(
        self, max_ast_size: Union[int, Literal["inf"]] = 50
    ) -> "DecompLTLSpec":
        return self.condense_assumptions(max_ast_size=max_ast_size).condense_guaranties(
            max_ast_size=max_ast_size
        )

    def deduce_inputs(self, inputs: Optional[List[str]] = None) -> List[str]:
        new_inputs = []
        for p in self.guarantees + self.assumptions:
            new_inputs = new_inputs + p.deduce_inputs(inputs)
        return sorted(list(set(new_inputs)))

    def deduce_outputs(self, outputs: Optional[List[str]] = None) -> List[str]:
        new_outputs = []
        for p in self.guarantees + self.assumptions:
            new_outputs = new_outputs + p.deduce_outputs(outputs)
        return sorted(list(set(new_outputs)))

    def reset_inputs(self, inputs: Optional[List[str]] = None):
        new_inputs = []
        if inputs is None:
            inputs = self.inputs
        for p in self.guarantees + self.assumptions:
            p.reset_inputs(inputs)
            new_inputs = new_inputs + p.inputs
        self.inputs = sorted(list(set(new_inputs)))

    def reset_outputs(self, outputs: Optional[List[str]] = None):
        new_outputs = []
        if outputs is None:
            outputs = self.outputs
        for p in self.guarantees + self.assumptions:
            p.reset_outputs(outputs)
            new_outputs = new_outputs + p.outputs
        self.outputs = sorted(list(set(new_outputs)))

    @classmethod
    def from_bosy_file(cls, filepath: str) -> "DecompLTLSpec":
        with open(filepath, "r") as spec_file:
            return cls.from_bosy_str(
                bosy_str=spec_file.read(),
                name=ntpath.basename(filepath)[:-5]
                if ntpath.basename(filepath).endswith(".json")
                else (
                    ntpath.basename(filepath)[:-4]
                    if ntpath.basename(filepath).endswith(".ltl")
                    else ntpath.basename(filepath)
                ),
            )

    @classmethod
    def from_bosy_str(cls, bosy_str: str, name: Optional[str] = None) -> "DecompLTLSpec":
        spec_dict = json.loads(bosy_str)
        if name is not None:
            spec_dict["name"] = name
        return cls.from_dict(spec_dict)

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["inputs", "outputs", "assumptions", "guarantees", "name", "notation", "semantics"]

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "DecompLTLSpec":
        return cls(
            assumptions=LTLAssumptions.from_csv_fields(fields),
            guarantees=LTLGuarantees.from_csv_fields(fields),
            inputs=fields["inputs"].split(",") if fields["inputs"] != "" else None,
            outputs=fields["outputs"].split(",") if fields["outputs"] != "" else None,
            name=fields.get("name", None),
            semantics=fields.get("semantics", None),
        )

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> "DecompLTLSpec":
        return cls(
            assumptions=LTLAssumptions.from_dict(d),
            guarantees=LTLGuarantees.from_dict(d),
            inputs=d["inputs"],
            outputs=d["outputs"],
            name=d.get("name", None),
            semantics=d.get("semantics", None),
        )

    @classmethod
    def from_pb2_DecompLTLSpecification(
        cls, pb2_DecompLTLSpecification, **kwargs
    ) -> "DecompLTLSpec":
        inputs = [str(i) for i in pb2_DecompLTLSpecification.inputs]
        outputs = [str(o) for o in pb2_DecompLTLSpecification.outputs]

        assumptions = LTLAssumptions(
            sub_exprs=[
                LTLSpec.from_dict(
                    {
                        "formula": a.formula,
                        "inputs": inputs,
                        "outputs": outputs,
                        "notation": a.notation,
                    },
                    **kwargs,
                )
                for a in pb2_DecompLTLSpecification.assumptions
            ],
            inputs=inputs,
            outputs=outputs,
        )

        guarantees = LTLGuarantees(
            sub_exprs=[
                LTLSpec.from_dict(
                    {
                        "formula": g.formula,
                        "inputs": inputs,
                        "outputs": outputs,
                        "notation": g.notation,
                    },
                    **kwargs,
                )
                for g in pb2_DecompLTLSpecification.guarantees
            ],
            inputs=inputs,
            outputs=outputs,
        )

        return cls(assumptions=assumptions, guarantees=guarantees, inputs=inputs, outputs=outputs)

    @staticmethod
    def comp_ast_pair(ast1: BinaryAST, ast2: BinaryAST) -> BinaryAST:
        if ast1 is None and ast2 is None:
            return None
        if ast1 is None:
            return ast2
        if ast2 is None:
            return ast1
        return BinaryAST("->", ast1, ast2)

    @staticmethod
    def comp_str_pair(str1: str, str2: str, notation: str = None) -> str:
        strs = [s for s in ["(" + str1 + ")", "(" + str2 + ")"] if s != ""]
        if notation == "infix" or notation == "infix-no-pars":
            return " -> ".join(strs)
        elif notation == "prefix":
            return "-> " * max(len(strs) - 1, 0) + " ".join(strs)
        else:
            raise ValueError(f"Unknown notation {notation}")

    @staticmethod
    def comp_token_list_pair(
        token_list1: List[str], token_list2: List[str], notation: str = None
    ) -> List[str]:
        if token_list1 == []:
            return token_list2
        if token_list2 == []:
            return token_list1
        if notation == "infix":
            return ["("] + token_list1 + [")"] + ["->"] + ["("] + token_list2 + [")"]
        elif notation == "infix-no-pars":
            return token_list1 + ["->"] + token_list2
        elif notation == "prefix":
            return ["->"] + token_list1 + token_list2
        else:
            raise ValueError(f"Unknown notation {notation}")

    @staticmethod
    def _condense_properties(
        properties: LTLProperties, max_ast_size: Union[int, Literal["inf"]] = 50
    ):
        acc_len = 0
        condense_list = []
        condensed = []
        last = False

        properties = deepcopy(properties)

        while len(properties) > 0:
            for i in range(len(properties)):
                size = properties[i].size()
                last = i == len(properties) - 1
                if max_ast_size != "inf" and acc_len == 0:
                    acc_len = acc_len + size
                    condense_list.append(properties.pop(i))
                    break
                if max_ast_size == "inf" or (acc_len + size + 1 <= max_ast_size):
                    acc_len = acc_len + size + 1
                    condense_list.append(properties.pop(i))
                    break
            if len(condense_list) > 0 and last:
                condensed.append(LTLProperties(sub_exprs=condense_list).condense())
                acc_len = 0
                condense_list = []
        return condensed
