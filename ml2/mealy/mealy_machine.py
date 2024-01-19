"""Mealy machine"""

import hashlib
import re
from typing import Dict, List, Optional

from ..datasets.utils import from_csv_str, to_csv_str
from ..dtypes import CSV, Hashable
from ..prop.prop_formula import PropFormula


class HoaHeader(Hashable):
    format_version: str
    num_states: int
    aps: List[str]
    controllable_aps_num: Optional[List[int]]
    init_state: int
    alias: Optional[str]
    acceptance: Optional[str]
    acc_name: Optional[str]
    tool: Optional[List[str]]
    name: Optional[str]
    properties: List[str]
    remainder: Dict

    def __init__(
        self,
        format_version: str,
        num_states: int,
        aps: List[str],
        init_state: int,
        controllable_aps_num: Optional[List[int]] = None,
        alias: Optional[str] = None,
        acceptance: Optional[str] = None,
        acc_name: Optional[str] = None,
        tool: Optional[List[str]] = None,
        name: Optional[str] = None,
        properties: List[str] = [],
        remainder: Dict = {},
    ):
        self.format_version = format_version
        self.num_states = num_states
        self.aps = aps
        self.controllable_aps_num = controllable_aps_num
        self.init_state = init_state
        self.alias = alias
        self.acceptance = acceptance
        self.acc_name = acc_name
        self.tool = tool
        self.name = name
        self.properties = properties
        self.remainder = remainder

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.format_version,
                        self.num_states,
                        sorted(self.aps) if self.aps is not None else None,
                        sorted(self.controllable_aps)
                        if self.controllable_aps is not None
                        else None,
                        self.init_state,
                        self.alias,
                        self.acceptance,
                        self.acc_name,
                        sorted(self.tool) if self.tool is not None else None,
                        self.name,
                        sorted(self.properties) if self.properties is not None else None,
                        sorted(self.remainder) if self.remainder is not None else None,
                    )
                ).encode()
            ).hexdigest(),
            16,
        )

    @property
    def ap_num(self) -> int:
        return len(self.aps)

    @property
    def inputs(self) -> List[str]:
        return list(filter(lambda x: x[0] == "i", self.aps))

    @property
    def outputs(self) -> List[str]:
        return list(filter(lambda x: x[0] == "o", self.aps))

    @property
    def controllable_aps(self) -> Optional[List[str]]:
        return (
            list(self.aps[int(ca)] for ca in self.controllable_aps_num)
            if self.controllable_aps_num
            else None
        )

    @classmethod
    def from_str(cls, header_str: str) -> "HoaHeader":
        header_str = header_str.replace('""', '"')
        header_dict = {}
        for r in (l.split(": ") for l in header_str.splitlines()):
            header_dict[r[0]] = r[1]
        ap_string = header_dict.pop("AP").replace('"', "")
        aps = ap_string[re.search("[0-9]+", ap_string).span()[1] + 1 :].split(" ")
        assert int(re.findall("[0-9]+", ap_string)[0]) == len(aps)
        return cls(
            format_version=header_dict.pop("HOA"),
            num_states=int(header_dict.pop("States")),
            aps=aps,
            controllable_aps_num=list(
                int(i) for i in header_dict.pop("controllable-AP").split(" ")
            )
            if "controllable-AP" in header_dict.keys()
            else None,
            init_state=int(header_dict.pop("Start")),
            alias=header_dict.pop("Alias") if "Alias" in header_dict.keys() else None,
            acceptance=(
                header_dict.pop("Acceptance") if "Acceptance" in header_dict.keys() else None
            ),
            acc_name=header_dict.pop("acc-name") if "acc-name" in header_dict.keys() else None,
            tool=header_dict.pop("tool").split(" ") if "tool" in header_dict.keys() else None,
            name=header_dict.pop("name") if "name" in header_dict.keys() else None,
            properties=(
                header_dict.pop("properties").split(" ")
                if "properties" in header_dict.keys()
                else []
            ),
            remainder=header_dict,
        )

    @classmethod
    def mealy_header(
        cls,
        num_states: int,
        inputs: List[str],
        outputs: List[str],
        init_state: int,
    ) -> "HoaHeader":
        return cls(
            format_version="v1",
            num_states=num_states,
            aps=inputs + outputs,
            init_state=init_state,
            acceptance="0 t",
            acc_name="all",
            properties=["trans-labels", "explicit-labels", "state-acc", "deterministic"],
        )

    def to_str(self, realizable: bool) -> str:
        return "HOA: {}\nStates: {}\nStart: {}\nAP: {} {}\nacc-name: {}\nAcceptance: {}\nproperties: {}\ncontrollable-AP: {}\n{}".format(
            self.format_version,
            str(self.num_states),
            str(self.init_state),
            str(self.ap_num),
            " ".join('"{}"'.format(ap) for ap in self.aps),
            self.acc_name,
            self.acceptance,
            " ".join(self.properties),
            " ".join(
                [
                    str(x[1])
                    for x in (
                        filter(
                            lambda x: x[0][0] == "o",
                            [(self.aps[i], i) for i in range(self.ap_num)],
                        )
                    )
                ]
                if realizable
                else [
                    str(x[1])
                    for x in (
                        filter(
                            lambda x: x[0][0] == "i",
                            [(self.aps[i], i) for i in range(self.ap_num)],
                        )
                    )
                ]
            )
            if self.controllable_aps_num is None
            else " ".join([str(i) for i in self.controllable_aps_num]),
            "\n".join((key + ": " + self.remainder[key]) for key in self.remainder.keys())
            + ("\n" if self.remainder else ""),
        )


class Condition(PropFormula):
    @classmethod
    def from_hoa_str(cls, cond: str, aps_list: List[str], notation: str = "infix") -> "Condition":
        aps = [aps_list[int(match)] for match in re.findall("[0-9]+", cond)]
        formula = ""
        for el in re.split("[0-9]+", cond):
            formula = formula + el
            if aps:
                formula = formula + aps.pop(0)
        return cls(formula=formula, notation=notation)

    def to_hoa_str(self, aps_list: List[str], notation: str = "infix") -> str:
        cond = self.to_str(notation=notation, space=False)
        for i in range(len(aps_list)):
            splitted = re.split(aps_list[i], cond)
            cond = "{}".format(str(i)).join(splitted)
        return cond


class Transition(Hashable):
    src: int
    cond: Condition
    dst: int

    def __init__(self, src: int, dst: int, cond: Condition):
        self.src = src
        self.dst = dst
        self.cond = cond

    @classmethod
    def from_str(cls, src: int, edge_str: str, notation: str = "infix") -> "Transition":
        return cls(
            src=src,
            dst=int("".join(reversed(re.findall("[0-9]+", "".join(reversed(edge_str)))[0]))),
            cond=Condition(
                formula=edge_str[
                    1 : -re.search("[0-9]+", "".join(reversed(edge_str))).span()[1] - 2
                ],
                notation=notation,
            ),
        )

    @classmethod
    def from_hoa_str(cls, src: int, edge_str: str, aps: List[str]) -> "Transition":
        return cls(
            src=src,
            dst=int("".join(reversed(re.findall("[0-9]+", "".join(reversed(edge_str)))[0]))),
            cond=Condition.from_hoa_str(
                edge_str[1 : -re.search("[0-9]+", "".join(reversed(edge_str))).span()[1] - 2],
                aps_list=aps,
            ),
        )

    def to_str(self, components: List[str] = ["cond", "dst"], notation: str = "infix") -> str:
        res = ""
        if "src" in components:
            res = res + "{} ".format(str(self.src))
        if "cond" in components:
            res = res + "[{}]".format(self.cond.to_str(notation=notation))
        if "dst" in components:
            res = res + " {}".format(str(self.dst))
        return res

    def to_hoa_str(
        self,
        aps: List[str],
        components: List[str] = ["cond", "dst"],
    ) -> str:
        res = ""
        if "src" in components:
            res = res + "{} ".format(str(self.src))
        if "cond" in components:
            res = res + "[{}]".format(self.cond.to_hoa_str(aps_list=aps))
        if "dst" in components:
            res = res + " {}".format(str(self.dst))
        return res

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(str((self.src, self.dst, self.cond.cr_hash)).encode()).hexdigest(),
            16,
        )


class MealyMachine(CSV):
    def __init__(
        self,
        header: HoaHeader,
        transitions: Optional[List[Transition]] = None,
    ):
        self.header = header
        self.transitions = transitions if transitions is not None else []

    @property
    def inputs(self) -> List[str]:
        return self.header.inputs

    @property
    def outputs(self) -> List[str]:
        return self.header.outputs

    @property
    def controllable_aps(self) -> Optional[List[str]]:
        return self.header.controllable_aps

    @property
    def num_states(self) -> int:
        return self.header.num_states

    @property
    def num_edges(self) -> int:
        return len(self.transitions)

    @classmethod
    def from_hoa(cls, hoa: str, **kwargs) -> "MealyMachine":
        transitions = []
        header = HoaHeader.from_str(hoa.split("--BODY--")[0])
        body = hoa.split("--BODY--")[1][1:-8]
        for row in body.split("State: ")[1:]:
            src = int(row[0 : re.search("[0-9]+", row).span()[1]])
            for edge in row[re.search("[0-9]+", row).span()[1] :].splitlines()[1:]:
                transitions.append(Transition.from_hoa_str(src, edge, header.aps))
        return MealyMachine(header, transitions)

    def to_hoa(self, realizable: bool, **kwargs) -> str:
        transitions_str = ""
        for i in range(self.num_states):
            transitions_str = transitions_str + "State: {}\n".format(i)
            transitions_str = transitions_str + "".join(
                t.to_hoa_str(components=["cond", "dst"], aps=self.header.aps) + "\n"
                for t in (filter(lambda t: t.src == i, self.transitions))
            )
        return "{}--BODY--\n{}--END--".format(
            self.header.to_str(realizable=realizable), transitions_str
        )

    def _to_csv_fields(self, realizable: bool, **kwargs) -> Dict[str, str]:
        return {
            "mealy": to_csv_str(self.to_hoa(realizable=realizable)),
        }

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> "MealyMachine":
        return cls.from_hoa(from_csv_str(fields["mealy"]))

    # @property
    # def cr_hash(self) -> int:
    #     return int(
    #         hashlib.sha3_224(
    #             str((self.header.cr_hash, sorted([t.cr_hash for t in self.transitions]))).encode()
    #         ).hexdigest(),
    #         16,
    #     )

    # def __eq__(self, __o: object) -> bool:
    #     if not isinstance(__o, MealyMachine):
    #         return False
    #     else:
    #         return self.cr_hash == __o.cr_hash
