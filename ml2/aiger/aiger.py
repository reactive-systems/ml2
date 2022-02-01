"""AIGER circuit class based on https://github.com/mvcisback/py-aiger/blob/main/aiger/parser.py"""

import re


class Header:
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


class Latch:
    def __init__(self, lit: int, next_lit: int):
        self.lit = lit
        self.next_lit = next_lit

    def __str__(self):
        return f"{self.lit} {self.next_lit}"


class And:
    def __init__(self, lit: int, arg1: int, arg2: int):
        self.lit = lit
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return f"{self.lit} {self.arg1} {self.arg2}"


class Symbol:
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


class Circuit:
    def __init__(
        self,
        header=None,
        inputs=None,
        latches=None,
        outputs=None,
        ands=None,
        symbols=None,
        comments=None,
    ):
        self.header = header
        self.inputs = inputs if inputs else []
        self.latches = latches if latches else []
        self.outputs = outputs if outputs else []
        self.ands = ands if ands else []
        self.symbols = symbols if symbols else []
        self.comments = comments if comments else []

    @property
    def max_var_id(self):
        lit = 0
        if self.inputs:
            lit = max(self.inputs)
        components = self.latches + self.ands
        if components:
            lit = max(lit, max([x.lit for x in components]))
        return lit // 2

    @property
    def num_inputs(self):
        return len(self.inputs)

    @property
    def num_latches(self):
        return len(self.latches)

    @property
    def num_outputs(self):
        return len(self.outputs)

    @property
    def num_ands(self):
        return len(self.ands)

    @property
    def input_var_ids(self):
        return [i // 2 for i in self.inputs]

    @property
    def latch_var_ids(self):
        return [l.lit // 2 for l in self.latches]

    @property
    def output_var_ids(self):
        return [o // 2 for o in self.outputs]

    @property
    def and_var_ids(self):
        return [a.lit // 2 for a in self.ands]

    def get_latch_by_idx(self, idx):
        for latch in self.latches:
            if latch.lit // 2 == idx:
                return latch
        return None

    def __str__(self):
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


HEADER_PATTERN = re.compile(r"aag (\d+) (\d+) (\d+) (\d+) (\d+)")


def parse_header(line, state):
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


IO_PATTERN = re.compile(r"(\d+)")


def parse_input(line, state):
    match = IO_PATTERN.fullmatch(line)
    if not match or state.num_inputs >= state.header.num_inputs:
        return False
    lit = int(line)
    state.inputs.append(lit)
    return True


def parse_output(line, state):
    match = IO_PATTERN.fullmatch(line)
    if not match or state.num_outputs >= state.header.num_outputs:
        return False
    lit = int(line)
    state.outputs.append(lit)
    return True


LATCH_PATTERN = re.compile(r"(\d+) (\d+)")


def parse_latch(line, state):
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

    state.latches.append(Latch(lit, next_lit))
    return True


AND_PATTERN = re.compile(r"(\d+) (\d+) (\d+)")


def parse_and(line, state):
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


SYM_PATTERN = re.compile(r"([ilo])(\d+) (.*)")


def parse_symbol(line, state):
    match = SYM_PATTERN.fullmatch(line)
    if not match:
        return False
    kind, idx, name = match.groups()
    state.symbols.append(Symbol(kind, idx, name))
    return True


def parse_comment(line, state):
    if state.comments:
        state.comments.append(line.restrip())
    elif line.rstrip() == "c":
        state.comments = []
    else:
        return False
    return True


DEFAULT_COMPONENTS = ["header", "inputs", "latches", "outputs", "ands", "symbols", "comments"]


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


def parse(circuit: str, components=None, state=None):
    if not components:
        components = DEFAULT_COMPONENTS

    if not state:
        state = Circuit()

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


def parse_no_header(circuit: str, num_inputs: int, num_outputs: int, components=None):
    state = Circuit()

    state.header = Header(None, num_inputs, None, num_outputs, None)

    parse(circuit, components, state)

    state.header.max_var_id = state.max_var_id
    state.header.num_latches = state.num_latches
    state.header.num_ands = state.num_ands

    return state
