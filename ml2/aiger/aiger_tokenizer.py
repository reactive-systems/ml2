"""AIGER circuit to sequence tokenizer"""

from typing import List, Optional, Type

from ..tokenizers import SeqToSeqTokenizer
from .aiger_circuit import AIGERCircuit, Symbol

NEWLINE_TOKEN = "<n>"
SEPARATOR_TOKEN = "<sep>"
COMPLEMENT_TOKEN = "<c>"
LATCH_TOKEN = "<l>"
REALIZABLE_TOKEN = "<r>"
UNREALIZABLE_TOKEN = "<u>"


class AIGERToSeqTokenizer(SeqToSeqTokenizer):
    def __init__(
        self,
        components: List[str] = None,
        dtype: Type[AIGERCircuit] = AIGERCircuit,
        inputs: List[str] = None,
        outputs: List[str] = None,
        unfold_negations: bool = False,
        unfold_latches: bool = False,
        ints_as_chars: bool = False,
        **kwargs,
    ):
        self.components = (
            components
            if components is not None
            else ["header", "inputs", "latches", "outputs", "ands", "symbols"]
        )

        self.inputs = inputs
        self.outputs = outputs
        self.unfold_negations = unfold_negations
        self.unfold_latches = unfold_latches
        self.ints_as_chars = ints_as_chars
        super().__init__(dtype=dtype, **kwargs)

    def encode_tokens(self, data: AIGERCircuit, **kwargs) -> List[str]:
        aiger = data
        tokens = []

        latch_var_ids = aiger.latch_var_ids

        for component in self.components:
            if component == "header":
                header_ints = str(aiger.header).split(" ")[1:]

                if self.unfold_latches:
                    header_ints[2] = "0"

                if self.ints_as_chars:
                    header_ints = [
                        elem
                        for string in header_ints
                        for elem in (list(string) + [SEPARATOR_TOKEN])
                    ][:-1]

                tokens.extend(header_ints)
                tokens.append(NEWLINE_TOKEN)
            elif component == "latches" and self.unfold_latches:
                continue
            else:
                for elem in getattr(aiger, component):
                    str_lits = str(elem).split(" ")
                    if self.ints_as_chars:
                        str_lits = [
                            elem
                            for string in str_lits
                            for elem in (list(string) + [SEPARATOR_TOKEN])
                        ][:-1]
                    seen_latch_ids = []

                    if self.unfold_negations or self.unfold_latches:
                        if self.ints_as_chars:
                            raise NotImplementedError  # could work but has not been checked

                        unfolded_str_lits = []

                        for i, str_lit in enumerate(str_lits):
                            lit = int(str_lit)

                            if self.unfold_latches:
                                if (component == "ands" and i > 0) or component == "outputs":
                                    latch_lit = lit
                                    latch_id = latch_lit // 2

                                    while (
                                        latch_id in latch_var_ids
                                        and latch_id not in seen_latch_ids
                                    ):
                                        seen_latch_ids.append(latch_lit // 2)
                                        if latch_lit % 2 == 1:
                                            unfolded_str_lits.append(COMPLEMENT_TOKEN)
                                        latch = aiger.get_latch_by_idx(latch_lit // 2)
                                        unfolded_str_lits.append(LATCH_TOKEN)
                                        latch_lit = latch.next_lit

                                    lit = latch_lit

                            if self.unfold_negations:
                                if lit % 2 == 1:
                                    unfolded_str_lits.append(COMPLEMENT_TOKEN)
                                lit = lit // 2

                            unfolded_str_lits.append(str(lit))

                        str_lits = unfolded_str_lits

                    tokens.extend(str_lits)
                    tokens.append(NEWLINE_TOKEN)

        # remove last newline token
        return tokens[:-1]

    def decode_tokens(
        self,
        tokens: List[str],
        realizable: bool = True,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        num_inputs: Optional[int] = None,
        num_outputs: Optional[int] = None,
        **kwargs,
    ) -> AIGERCircuit:
        """Priority how inputs and outputs. symbol table and header are reconstructed:
        1. Header and symbol table are encoded. All other related arguments are ignored.
        2. Only symbol table is encoded (no header). Header will be reconstructed with given **number** of inputs and outputs through
            1. inputs / output numbers that is given to the decoding method as argument (num_inputs / num_outputs)
            2. length of inputs / output list that is given to the decoding method as argument (inputs / outputs)
            3. length of input / output list the tokenizer has (self.inputs / self.outputs)
        3. Only header is encoded (no symbol table). Symbol table will be reconstructed with subset of given list of inputs and outputs (order important!!) through
            1. inputs / output list that is given to the decoding method as argument (inputs / outputs)
            2. input / output list the tokenizer has (self.inputs / self.outputs)
            3. Generic i0...iN / o0...oN list with N extracted from header.
        4. No symbol table, no header is encoded. Header and symbol table will be reconstructed with given list of inputs and outputs (order important!!) through
            1. inputs / output list that is given to the decoding method as argument (inputs / outputs)
            2. input / output list the tokenizer has (self.inputs / self.outputs)
            3. Generic i0...iN / o0...oN list with N given as arguments to the decoding method (num_inputs / num_outputs).

        Note that when using generic lists, training data has to use the same generic list in the same order!
        """

        components = list(self.components)

        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs

        if "header" not in self.components:
            if num_inputs is None:
                num_inputs = len(inputs) if inputs is not None else None
            if num_outputs is None:
                num_outputs = len(outputs) if outputs is not None else None
            assert num_outputs is not None and num_inputs is not None

        if "symbols" in self.components:
            inputs = None
            outputs = None

        if self.unfold_latches or self.unfold_negations:
            raise NotImplementedError
            # The following code might work fine if header or symbol table do not need to be reconstructed in a complicated way. Also might not work with self.ints_as_chars
            sequence = " ".join(tokens)
            lines = sequence.split(f" {NEWLINE_TOKEN} ")
            header = None

            if "header" in self.components:
                try:
                    header = lines.pop(0)
                    num_inputs = int(header.split(" ")[1])
                    num_outputs = int(header.split(" ")[3])
                except ValueError:
                    raise Exception("Could not decode header")

            folded_tokens = []
            next_latch_lit = (num_inputs + 1) * 2
            latches = {}
            for line in lines:
                line_tokens = []
                line_split = line.split(" ")
                last_lit = None

                for str_lit in reversed(line_split):
                    if str_lit == LATCH_TOKEN:
                        if last_lit is None:
                            raise Exception("Illegal latch token")
                        if last_lit not in latches:
                            latches[last_lit] = next_latch_lit
                            next_latch_lit += 2
                        last_lit = latches[last_lit]
                    elif str_lit == COMPLEMENT_TOKEN:
                        if last_lit is None:
                            raise Exception("Illegal negation token")
                        last_lit += 1
                    else:
                        try:
                            lit = int(str_lit)
                        except ValueError:
                            raise Exception("Illegal token")
                        else:
                            if self.unfold_negations:
                                lit = lit * 2
                            if last_lit is not None:
                                line_tokens.insert(0, str(last_lit))
                            last_lit = lit

                line_tokens.insert(0, str(last_lit))
                folded_tokens.extend(line_tokens)
                folded_tokens.append(NEWLINE_TOKEN)

            for next_lit, lit in latches.items():
                folded_tokens.extend([str(lit), str(next_lit), NEWLINE_TOKEN])
                components.remove("latches")
                components.append("latches")

            folded_tokens = folded_tokens[:-1]

            if "header" in self.components:
                header_ints = header.split(" ")
                if self.unfold_latches:
                    try:
                        header_ints[2] = str(len(latches))
                    except IndexError:
                        raise Exception("Could not decode header")
                folded_tokens = header_ints + [NEWLINE_TOKEN] + folded_tokens

            sequence = " ".join(folded_tokens)

        else:
            if self.ints_as_chars:
                tokens = [token if token != SEPARATOR_TOKEN else " " for token in tokens]
                sequence = "".join(tokens)
                sequence = sequence.replace(NEWLINE_TOKEN, " " + NEWLINE_TOKEN + " ")
            else:
                sequence = " ".join(tokens)

        sequence = sequence.replace(NEWLINE_TOKEN, "\n")
        sequence = sequence.replace(" \n ", "\n")

        if "header" not in self.components:
            assert num_inputs is not None and num_outputs is not None
            aiger = AIGERCircuit.from_str_without_header(
                circuit=sequence,
                num_inputs=num_inputs if realizable else num_outputs,
                num_outputs=num_outputs if realizable else num_inputs,
                components=components,
            )
        else:
            sequence = "aag " + sequence
            aiger = AIGERCircuit.from_str(circuit=sequence, components=components)
            num_inputs = aiger.num_inputs if realizable else aiger.num_outputs
            num_outputs = aiger.num_outputs if realizable else aiger.num_inputs

        if "symbols" not in self.components:
            assert num_inputs is not None and num_outputs is not None
            inputs = ["i" + str(i) for i in range(num_inputs)] if inputs is None else inputs
            outputs = ["o" + str(i) for i in range(num_outputs)] if outputs is None else outputs

            symbols = [
                Symbol("i", i, inputs[i] if realizable else outputs[i])
                for i in range(num_inputs if realizable else num_outputs)
            ]
            symbols.extend([Symbol("l", i, f"l{i}") for i in range(aiger.num_latches)])
            symbols.extend(
                [
                    Symbol("o", i, outputs[i] if realizable else inputs[i])
                    for i in range(num_outputs if realizable else num_inputs)
                ]
            )
            aiger.symbols = symbols

        return aiger

    def sort_tokens(self, tokens: List[str]) -> None:
        tokens.sort()
        tokens.sort(key=len)

    def vocabulary_filename(self) -> str:
        return "aiger-vocab" + super().vocabulary_filename()
