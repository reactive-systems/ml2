"""AIGER circuit encoder"""

import tensorflow as tf
from typing import List

from .aiger import parse, parse_no_header, Symbol
from ..data.encoder import SeqEncoder
from ..data.vocabulary import Vocabulary

NEWLINE_TOKEN = "<n>"
COMPLEMENT_TOKEN = "<c>"
LATCH_TOKEN = "<l>"
REALIZABLE_TOKEN = "<r>"
UNREALIZABLE_TOKEN = "<u>"


class AIGERSequenceEncoder(SeqEncoder):
    def __init__(
        self,
        start: bool,
        eos: bool,
        pad: int,
        components: List[str] = None,
        encode_start: bool = True,
        encode_realizable: bool = False,
        inputs: List[str] = None,
        outputs: List[str] = None,
        unfold_negations: bool = False,
        unfold_latches: bool = False,
        vocabulary: Vocabulary = None,
        tf_dtype: tf.DType = tf.int32,
    ):
        """
        inputs, outputs: only required if components does not contain header or symbols
        """
        self.components = components if components else ["inputs", "latches", "outputs", "ands"]
        self.encode_realizable = encode_realizable
        self.inputs = inputs
        self.outputs = outputs
        self.realizable = True
        self.unfold_negations = unfold_negations
        self.unfold_latches = unfold_latches
        super().__init__(start, eos, pad, encode_start, vocabulary, tf_dtype)

    @property
    def circuit(self):
        return self.sequence

    def encode(self, sequence) -> bool:
        self.realizable = "i0 i0" in sequence
        return super().encode(sequence)

    def lex(self) -> bool:
        self.tokens = []

        if self.encode_realizable:
            if self.realizable:
                self.tokens.append(REALIZABLE_TOKEN)
            else:
                self.tokens.append(UNREALIZABLE_TOKEN)

        try:
            aiger = parse(self.circuit)
        except ValueError as err:
            self.error = err
            return False

        latch_var_ids = aiger.latch_var_ids

        for component in self.components:
            if component == "header":
                header_ints = str(aiger.header).split(" ")[1:]

                if self.unfold_latches:
                    header_ints[2] = "0"

                self.tokens.extend(header_ints)
                self.tokens.append(NEWLINE_TOKEN)
            elif component == "latches" and self.unfold_latches:
                continue
            else:
                for elem in getattr(aiger, component):
                    str_lits = str(elem).split(" ")
                    seen_latch_ids = []

                    if self.unfold_negations or self.unfold_latches:
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

                    self.tokens.extend(str_lits)
                    self.tokens.append(NEWLINE_TOKEN)

        # remove last newline token
        self.tokens = self.tokens[:-1]

        return True

    def decode(self, ids: List[int], realizable: bool = True) -> bool:
        success = super().decode(ids)
        components = list(self.components)

        if self.encode_realizable:
            realizable_token = self.sequence[:3]
            self.sequence = self.sequence[4:]
            if realizable_token not in (REALIZABLE_TOKEN, UNREALIZABLE_TOKEN):
                self.error = "First token not realizable token"
                return False
            else:
                if realizable_token == REALIZABLE_TOKEN:
                    self.realizable = True
                else:
                    self.realizable = False

        if "header" not in self.components or "symbols" not in self.components:
            num_inputs = len(self.inputs)
            num_outputs = len(self.outputs)

        if self.unfold_latches or self.unfold_negations:
            lines = self.sequence.split(f" {NEWLINE_TOKEN} ")
            header = None

            if "header" in self.components:
                try:
                    header = lines.pop(0)
                    num_inputs = int(header.split(" ")[1])
                except ValueError:
                    self.error = "Could not decode header"
                    return False

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
                            self.error = "Illegal latch token"
                            return False
                        if last_lit not in latches:
                            latches[last_lit] = next_latch_lit
                            next_latch_lit += 2
                        last_lit = latches[last_lit]
                    elif str_lit == COMPLEMENT_TOKEN:
                        if last_lit is None:
                            self.error = "Illegal negation token"
                            return False
                        last_lit += 1
                    else:
                        try:
                            lit = int(str_lit)
                        except ValueError:
                            self.error = "Illegal token"
                            return False
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
                        self.error = "Could not decode header"
                        return False
                folded_tokens = header_ints + [NEWLINE_TOKEN] + folded_tokens

            self.sequence = " ".join(folded_tokens)

        self.sequence = self.sequence.replace(NEWLINE_TOKEN, "\n")
        self.sequence = self.sequence.replace(" \n ", "\n")
        if "header" not in self.components:
            try:
                aiger = parse_no_header(self.sequence, num_inputs, num_outputs, components)
            except ValueError as error:
                self.error = error
                return False
        else:
            self.sequence = "aag " + self.sequence
            try:
                aiger = parse(self.sequence, components)
            except ValueError as error:
                self.error = error
                return False
        if "symbols" not in self.components:
            symbols = [
                Symbol("i", i, self.inputs[i] if self.realizable else self.outputs[i])
                for i in range(num_inputs)
            ]
            symbols.extend([Symbol("l", i, f"l{i}") for i in range(aiger.num_latches)])
            symbols.extend(
                [
                    Symbol("o", i, self.outputs[i] if self.realizable else self.inputs[i])
                    for i in range(num_outputs)
                ]
            )
            aiger.symbols = symbols
        self.sequence = str(aiger)
        return success

    def sort_tokens(self, tokens: list) -> None:
        tokens.sort()
        tokens.sort(key=len)

    def vocabulary_filename(self) -> str:
        return "aiger-vocab" + super().vocabulary_filename()
