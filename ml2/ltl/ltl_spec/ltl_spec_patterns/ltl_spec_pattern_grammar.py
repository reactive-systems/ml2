"""Class representing grammar for LTL specification patterns"""

import argparse
from functools import lru_cache
from itertools import chain, product
import math

from .ltl_spec_patterns import *

DEFAULT_PARAMS = {
    "input_after_n_steps": {"max_n": 2},
    "input_for_n_steps": {"max_n": 2},
    "input_within_n_steps": {"max_n": 2},
    "output_after_n_steps": {"max_n": 2},
    "output_for_n_steps": {"max_n": 2},
    "output_within_n_steps": {"max_n": 2},
    "cnf_mutual_exclusion": {"min_num_vars": 2, "max_num_vars": 3},
    "dnf_mutual_exclusion": {"min_num_vars": 3, "max_num_vars": 3},
    "factorized_mutual_exclusion": {"min_num_vars": 3, "max_num_vars": 3},
}


class LTLSpecPatternGrammar:
    def __init__(self, params=DEFAULT_PARAMS):
        self.params = params

    def derive_all(self):
        # derives all possible patterns from grammar
        return self.input_output_patterns + self.output_output_patterns

    @property
    @lru_cache()
    def input_output_patterns(self):
        return self.input_output_guarantee

    @property
    @lru_cache()
    def output_output_patterns(self):
        return self.output_output_guarantee

    @property
    @lru_cache()
    def input_output_guarantee(self):
        return list(chain(self.response, self.precedence, self.correspondence))

    @property
    @lru_cache()
    def output_output_guarantee(self):
        return list(chain(self.mutual_exclusion, self.output_reactivity))

    @property
    @lru_cache()
    def response(self):
        obl_react = [
            Response(in_obl, out_react)
            for (in_obl, out_react) in product(self.input_obligation, self.output_reactivity)
        ]
        react_obl = [
            Response(in_react, out_obl)
            for (in_react, out_obl) in product(self.input_reactivity, self.output_obligation)
        ]
        return obl_react + react_obl

    @property
    @lru_cache()
    def precedence(self):
        obl_react = [
            Precedence(out_obl, in_react)
            for (out_obl, in_react) in product(self.output_obligation, self.input_reactivity)
        ]
        react_obl = [
            Precedence(out_react, in_obl)
            for (out_react, in_obl) in product(self.output_reactivity, self.input_obligation)
        ]
        return obl_react + react_obl

    @property
    @lru_cache()
    def correspondence(self):
        obl_react = [
            Correspondece(in_obl, out_react)
            for (in_obl, out_react) in product(self.input_obligation, self.output_reactivity)
        ]
        react_obl = [
            Correspondece(in_react, out_obl)
            for (in_react, out_obl) in product(self.input_reactivity, self.output_obligation)
        ]
        return obl_react + react_obl

    @property
    @lru_cache()
    def mutual_exclusion(self):
        return list(
            chain(
                self.cnf_mutual_exclusion,
                self.dnf_mutual_exclusion,
                self.factorized_mutual_exclusion,
            )
        )

    @property
    @lru_cache()
    def cnf_mutual_exclusion(self):
        min_num_vars = self.params["cnf_mutual_exclusion"]["min_num_vars"]
        max_num_vars = self.params["cnf_mutual_exclusion"]["max_num_vars"]
        return [
            CNFMutualExclusion([self.atomic_output for _ in range(num_vars)])
            for num_vars in range(min_num_vars, max_num_vars + 1)
        ]

    @property
    @lru_cache()
    def dnf_mutual_exclusion(self):
        min_num_vars = self.params["dnf_mutual_exclusion"]["min_num_vars"]
        max_num_vars = self.params["dnf_mutual_exclusion"]["max_num_vars"]
        return [
            DNFMutualExclusion([self.atomic_output for _ in range(num_vars)])
            for num_vars in range(min_num_vars, max_num_vars + 1)
        ]

    @property
    @lru_cache()
    def factorized_mutual_exclusion(self):
        min_num_vars = self.params["factorized_mutual_exclusion"]["min_num_vars"]
        max_num_vars = self.params["factorized_mutual_exclusion"]["max_num_vars"]
        cnf = [
            FactorizedCNFMutualExclusion([self.atomic_output for _ in range(num_vars)])
            for num_vars in range(min_num_vars, max_num_vars + 1)
        ]
        dnf = [
            FactorizedDNFMutualExclusion([self.atomic_output for _ in range(num_vars)])
            for num_vars in range(min_num_vars, max_num_vars + 1)
        ]
        return cnf + dnf

    @property
    @lru_cache()
    def input_reactivity(self):
        return list(chain(self.input_obligation, self.input_recurrence, self.input_persistence))

    @property
    @lru_cache()
    def input_obligation(self):
        return list(
            chain(
                self.input_after_n_steps,
                self.input_for_n_steps,
                self.input_within_n_steps,
                self.eventually_input,
                self.globally_input,
                self.input,
            )
        )

    @property
    @lru_cache()
    def output_reactivity(self):
        return list(chain(self.output_obligation, self.output_recurrence, self.output_persistence))

    @property
    @lru_cache()
    def output_obligation(self):
        return list(
            chain(
                self.output_after_n_steps,
                self.output_for_n_steps,
                self.output_within_n_steps,
                self.eventually_output,
                self.globally_output,
                self.output,
            )
        )

    @property
    @lru_cache()
    def input_after_n_steps(self):
        max_n = self.params["input_after_n_steps"]["max_n"]
        patterns = [AfterNSteps(input, n + 1) for (input, n) in product(self.input, range(max_n))]
        return patterns

    @property
    @lru_cache()
    def input_for_n_steps(self):
        max_n = self.params["input_for_n_steps"]["max_n"]
        return [ForNSteps(input, n + 1) for (input, n) in product(self.input, range(max_n))]

    @property
    @lru_cache()
    def input_within_n_steps(self):
        max_n = self.params["input_within_n_steps"]["max_n"]
        return [WithinNSteps(input, n + 1) for (input, n) in product(self.input, range(max_n))]

    @property
    @lru_cache()
    def eventually_input(self):
        patterns = [Eventually([input]) for input in self.input]
        patterns.append(Eventually([self.atomic_input, self.atomic_input]))
        return patterns

    @property
    @lru_cache()
    def globally_input(self):
        patterns = [Globally([input]) for input in self.input]
        patterns.append(Globally([self.atomic_input, self.atomic_input]))
        return patterns

    @property
    @lru_cache()
    def input_recurrence(self):
        patterns = [Recurrence([input]) for input in self.input]
        patterns.append(Recurrence([self.atomic_input, self.atomic_input]))
        return patterns

    @property
    @lru_cache()
    def input_persistence(self):
        patterns = [Persistence([input]) for input in self.input]
        patterns.append(Persistence([self.atomic_input, self.atomic_input]))
        return patterns

    @property
    @lru_cache()
    def output_after_n_steps(self):
        max_n = self.params["output_after_n_steps"]["max_n"]
        return [AfterNSteps(output, n + 1) for (output, n) in product(self.output, range(max_n))]

    @property
    @lru_cache()
    def output_for_n_steps(self):
        max_n = self.params["output_for_n_steps"]["max_n"]
        return [ForNSteps(output, n + 1) for (output, n) in product(self.output, range(max_n))]

    @property
    @lru_cache()
    def output_within_n_steps(self):
        max_n = self.params["output_within_n_steps"]["max_n"]
        return [WithinNSteps(output, n + 1) for (output, n) in product(self.output, range(max_n))]

    @property
    @lru_cache()
    def eventually_output(self):
        patterns = [Eventually([output]) for output in self.output]
        patterns.append(Eventually([self.atomic_output, self.atomic_output]))
        return patterns

    @property
    @lru_cache()
    def globally_output(self):
        patterns = [Globally([output]) for output in self.output]
        patterns.append(Globally([self.atomic_output, self.atomic_output]))
        return patterns

    @property
    @lru_cache()
    def output_recurrence(self):
        patterns = [Recurrence([output]) for output in self.output]
        patterns.append(Recurrence([self.atomic_output, self.atomic_output]))
        return patterns

    @property
    @lru_cache()
    def output_persistence(self):
        patterns = [Persistence([output]) for output in self.output]
        patterns.append(Persistence([self.atomic_output, self.atomic_output]))
        return patterns

    @property
    @lru_cache()
    def input(self):
        return [self.atomic_input, AndInput(), OrInput()]

    @property
    @lru_cache()
    def atomic_input(self):
        return AtomicInput()

    @property
    @lru_cache()
    def output(self):
        return [self.atomic_output, AndOutput(), OrOutput()]

    @property
    @lru_cache()
    def atomic_output(self):
        return AtomicOutput()


def factorial(start, steps):
    return int(math.factorial(start) / math.factorial(start - steps))


def grammar_cardinality(num_inputs, num_outputs):
    cardinality = 0
    for pattern in LTLSpecPatternGrammar().derive_all():
        cardinality += factorial(num_inputs, pattern.num_inputs) * factorial(
            num_outputs, pattern.num_outputs
        )
    return cardinality


def add_parser_args(parser):
    parser.add_argument(
        "--num-inputs", type=int, default=5, help="number of input atomic propositions"
    )
    parser.add_argument(
        "--num-outputs", type=int, default=5, help="number of output atomic propositions"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculating grammar cardinality")
    add_parser_args(parser)
    args = parser.parse_args()
    cardinality = grammar_cardinality(args.num_inputs, args.num_outputs)
    print(f"Grammar cardinality: {cardinality}")
