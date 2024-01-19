"""AIGER circuit class based on https://github.com/mvcisback/py-aiger/blob/main/aiger/parser.py"""

import hashlib
import re
from random import sample
from typing import Dict, List, Optional

from numpy.random import default_rng

from ..datasets.utils import from_csv_str, to_csv_str
from ..dtypes import CSVWithId, Hashable, Seq
from ..registry import register_type


class Header(Hashable):
    def __init__(
        self, max_var_id: int, num_inputs: int, num_latches: int, num_outputs: int, num_ands: int
    ):
        self.max_var_id = max_var_id
        self.num_inputs = num_inputs
        self.num_latches = num_latches
        self.num_outputs = num_outputs
        self.num_ands = num_ands

    def __str__(self):
        return (
            f"aag {self.max_var_id} {self.num_inputs} {self.num_latches} "
            f"{self.num_outputs} {self.num_ands}"
        )

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.max_var_id,
                        self.num_inputs,
                        self.num_latches,
                        self.num_outputs,
                        self.num_ands,
                    )
                ).encode()
            ).hexdigest(),
            16,
        )


HEADER_PATTERN = re.compile(r"aag (\d+) (\d+) (\d+) (\d+) (\d+)")


def parse_header(line: str, state):
    if state.header:
        return False
    match = HEADER_PATTERN.fullmatch(line)
    if not match:
        raise ValueError(f"Failed to parse aag header: {line}")

    try:
        ids = [int(idx) for idx in match.groups()]

        if any(x < 0 for x in ids):
            raise ValueError("Indicies must be positive")

        max_var_id, num_inputs, num_latches, num_outputs, num_ands = ids
        if num_inputs + num_latches + num_ands > max_var_id:
            raise ValueError(
                "Sum of number of inputs, latches and ands is greater than max variable index"
            )

        state.header = Header(max_var_id, num_inputs, num_latches, num_outputs, num_ands)

    except ValueError as exc:
        raise ValueError("Failed to parse aag header") from exc
    return True


class Latch(Hashable):
    def __init__(self, lit: int, next_lit: int, reset: Optional[int] = None):
        self.lit = lit
        self.next_lit = next_lit
        assert reset is None or reset == 0 or reset == 1 or reset == self.lit
        self.reset: Optional[int] = None

    def __str__(self):
        if self.reset is None:
            return f"{self.lit} {self.next_lit}"
        else:
            return f"{self.lit} {self.next_lit} {self.reset}"

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (self.lit, self.next_lit)
                    if self.reset is None or self.reset == 0
                    else (self.lit, self.next_lit, self.reset)
                ).encode()
            ).hexdigest(),
            16,
        )


LATCH_PATTERN = re.compile(r"(\d+) (\d+)(?: (\1|0|1))?")


def parse_latch(line: str, state):
    if state.header.num_latches and state.num_latches >= state.header.num_latches:
        return False

    match = LATCH_PATTERN.fullmatch(line)
    if not match:
        if state.header.num_latches:
            raise ValueError(f"Expecting a latch: {line}")
        return False

    groups = match.groups()
    lit = int(groups[0])
    next_lit = int(groups[1])
    reset = int(groups[2]) if groups[2] is not None else None

    state.latches.append(Latch(lit, next_lit, reset))
    return True


class And(Hashable):
    def __init__(self, lit: int, arg1: int, arg2: int):
        self.lit = lit
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return f"{self.lit} {self.arg1} {self.arg2}"

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.lit,
                        self.arg1,
                        self.arg2,
                    )
                ).encode()
            ).hexdigest(),
            16,
        )


AND_PATTERN = re.compile(r"(\d+) (\d+) (\d+)")


def parse_and(line: str, state):
    if state.header.num_ands and state.num_ands >= state.header.num_ands:
        return False

    match = AND_PATTERN.fullmatch(line)
    if not match:
        if state.header.num_ands:
            raise ValueError(f"Expecting an and: {line}")
        return False

    groups = match.groups()
    lit = int(groups[0])
    arg1 = int(groups[1])
    arg2 = int(groups[2])

    state.ands.append(And(lit, arg1, arg2))
    return True


class Symbol(Hashable):
    def __init__(
        self,
        kind: str,
        idx: int,
        name: str,
    ):
        self.kind = kind
        self.idx = idx
        self.name = name

    def __str__(self):
        return f"{self.kind}{self.idx} {self.name}"

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.kind,
                        self.idx,
                        self.name,
                    )
                ).encode()
            ).hexdigest(),
            16,
        )


SYM_PATTERN = re.compile(r"([ilo])(\d+) (.*)")


def parse_symbol(line: str, state):
    match = SYM_PATTERN.fullmatch(line)
    if not match:
        return False
    kind, idx, name = match.groups()
    state.symbols.append(Symbol(kind, idx, name))
    return True


IO_PATTERN = re.compile(r"(\d+)")


def parse_input(line: str, state):
    match = IO_PATTERN.fullmatch(line)
    if not match or state.num_inputs >= state.header.num_inputs:
        return False
    lit = int(line)
    state.inputs.append(lit)
    return True


def parse_output(line: str, state):
    match = IO_PATTERN.fullmatch(line)
    if not match or state.num_outputs >= state.header.num_outputs:
        return False
    lit = int(line)
    state.outputs.append(lit)
    return True


def parse_comment(line: str, state):
    if state.comments:
        state.comments.append(line.rstrip())
    elif line.rstrip() == "c":
        state.comments = ["c"]
    else:
        return False
    return True


@register_type
class AIGERCircuit(CSVWithId, Seq):
    def __init__(
        self,
        header: Header = None,
        inputs: List[int] = None,
        latches: List[Latch] = None,
        outputs: List[int] = None,
        ands: List[And] = None,
        symbols: List[Symbol] = None,
        comments: List[str] = None,
    ):
        self.header = header
        self.inputs = inputs if inputs else []
        self.latches = latches if latches else []
        self.outputs = outputs if outputs else []
        self.ands = ands if ands else []
        self.symbols = symbols if symbols else []
        self.comments = comments if comments else []

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, AIGERCircuit):
            return False
        else:
            return self.cr_hash == __o.cr_hash

    @property
    def cr_hash(self) -> int:
        return int(
            hashlib.sha3_224(
                str(
                    (
                        self.header.cr_hash,
                        sorted(self.inputs),
                        sorted(self.outputs),
                        sorted([l.cr_hash for l in self.latches]),
                        sorted([a.cr_hash for a in self.ands]),
                        sorted([s.cr_hash for s in self.symbols]),
                    )
                ).encode()
            ).hexdigest(),
            16,
        )

    def rename_aps(
        self,
        input_aps: List[str] = None,
        output_aps: List[str] = None,
        random: bool = True,
        random_weighted: Optional[Dict[str, float]] = None,
        realizable: bool = None,
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
            if realizable is None:
                raise ValueError("realizable must be given when renaming dict not set")
            num_inputs_adapted = self.num_inputs if realizable else self.num_outputs
            num_outputs_adapted = self.num_outputs if realizable else self.num_inputs
            if input_aps is None:
                input_aps = ["i" + str(i) for i in range(num_inputs_adapted)]
            if output_aps is None:
                output_aps = ["o" + str(i) for i in range(num_outputs_adapted)]
            if random:
                if random_weighted is None:
                    renamed_inputs = sample(input_aps, num_inputs_adapted)
                    renamed_outputs = sample(output_aps, num_outputs_adapted)
                else:
                    renamed_inputs = weighted_random(
                        input_aps, random_weighted=random_weighted, number=num_inputs_adapted
                    )
                    renamed_outputs = weighted_random(
                        output_aps, random_weighted=random_weighted, number=num_outputs_adapted
                    )
            else:
                renamed_inputs = input_aps[:num_inputs_adapted]
                renamed_outputs = output_aps[:num_outputs_adapted]

            if not realizable:
                h = renamed_outputs
                renamed_outputs = renamed_inputs
                renamed_inputs = h
            renaming = dict(
                zip(self.input_symbols + self.output_symbols, renamed_inputs + renamed_outputs)
            )

        for symbol in self.symbols:
            symbol.name = renaming[symbol.name] if symbol.name in renaming.keys() else symbol.name

        inv_renaming = {v: k for k, v in renaming.items()}
        return inv_renaming

    @property
    def max_var_id(self) -> int:
        lit = 0
        if self.inputs:
            lit = max(self.inputs)
        components = self.latches + self.ands
        if components:
            lit = max(lit, max([x.lit for x in components]))
        return lit // 2

    @property
    def input_symbols(self) -> List[str]:
        return [symbol.name for symbol in filter(lambda x: x.kind == "i", self.symbols)]

    @property
    def output_symbols(self) -> List[str]:
        return [symbol.name for symbol in filter(lambda x: x.kind == "o", self.symbols)]

    @property
    def num_inputs(self) -> int:
        return len(self.inputs)

    @property
    def num_latches(self) -> int:
        return len(self.latches)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def num_ands(self) -> int:
        return len(self.ands)

    @property
    def input_var_ids(self) -> List[int]:
        return [i // 2 for i in self.inputs]

    @property
    def latch_var_ids(self) -> List[int]:
        return [l.lit // 2 for l in self.latches]

    @property
    def output_var_ids(self) -> List[int]:
        return [o // 2 for o in self.outputs]

    @property
    def and_var_ids(self) -> List[int]:
        return [a.lit // 2 for a in self.ands]

    def get_latch_by_idx(self, idx: int) -> Optional[Latch]:
        for latch in self.latches:
            if latch.lit // 2 == idx:
                return latch
        return None

    def _to_csv_fields(self, **kwargs) -> Dict[str, str]:
        return {"circuit": to_csv_str(self.to_str())}

    def to_str(self, **kwargs) -> str:
        return "\n".join(
            [
                str(x)
                for x in [
                    self.header,
                    *self.inputs,
                    *self.latches,
                    *self.outputs,
                    *self.ands,
                    *self.symbols,
                    *self.comments,
                ]
            ]
        )

    def to_tokens(self, **kwargs) -> List[str]:
        tokens = []

        # header
        tokens.extend(str(self.header).split(" ")[1:])
        # new line token
        tokens.append("<n>")

        for component in ["inputs", "latches", "outputs", "ands", "symbols", "comments"]:
            for elem in getattr(self, component):
                str_lits = str(elem).split(" ")
                tokens.extend(str_lits)
                # new line token
                tokens.append("<n>")

        # remove last newline token
        return tokens[:-1]

    def reset_header(self) -> None:
        self.header = Header(
            max_var_id=self.max_var_id,
            num_inputs=self.num_inputs,
            num_ands=self.num_ands,
            num_latches=self.num_latches,
            num_outputs=self.num_outputs,
        )

    @classmethod
    def _from_csv_fields(cls, fields: Dict[str, str], **kwargs) -> Optional["AIGERCircuit"]:
        return (
            cls.from_str(circuit=from_csv_str(fields["circuit"]))
            if "circuit" in fields.keys()
            and fields["circuit"] != ""
            and fields["circuit"] != "None"
            else None
        )

    @classmethod
    def _csv_field_header(cls, **kwargs) -> List[str]:
        return ["circuit"]

    @classmethod
    def from_str(
        cls, circuit: str, components: List[str] = None, state: "AIGERCircuit" = None, **kwargs
    ) -> "AIGERCircuit":
        if not components:
            components = ["header", "inputs", "latches", "outputs", "ands", "symbols", "comments"]

        if state is None:
            state = AIGERCircuit()

        parsers = parser_seq(components)
        parser = next(parsers)

        lines = circuit.split("\n")
        for line in lines:
            while not parser(line, state):
                try:
                    parser = next(parsers)
                except StopIteration as exc:
                    raise ValueError(f"Could not parse line: {line}") from exc

        return state

    @classmethod
    def from_str_without_header(
        cls, circuit: str, num_inputs: int, num_outputs: int, components: List[str] = None
    ) -> "AIGERCircuit":
        state = AIGERCircuit()

        state.header = Header(
            max_var_id=None,
            num_inputs=num_inputs,
            num_latches=None,
            num_outputs=num_outputs,
            num_ands=None,
        )

        cls.from_str(circuit=circuit, components=components, state=state)

        state.header.max_var_id = state.max_var_id
        state.header.num_latches = state.num_latches
        state.header.num_ands = state.num_ands

        return state

    @classmethod
    def from_tokens(cls, tokens: List[str], **kwargs) -> "AIGERCircuit":
        circuit = "aag " + " ".join(tokens)
        circuit = circuit.replace(" <n> ", "\n")
        return cls.from_str(circuit=circuit)


def parser_seq(components):
    for component in components:
        yield {
            "header": parse_header,
            "inputs": parse_input,
            "latches": parse_latch,
            "outputs": parse_output,
            "ands": parse_and,
            "symbols": parse_symbol,
            "comments": parse_comment,
        }.get(component)
