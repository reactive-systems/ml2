"""LTL specification pattern data"""

import argparse
import json
import logging
import os

from numpy import random

from ...data import Data
from ..ltl_lexer import lex_ltl
from ...globals import LTL_SPEC_ALIASES, LTL_SPEC_BUCKET_DIR, LTL_SPEC_WANDB_PROJECT
from .ltl_spec_data import LTLSpecData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSpecPatternData(Data):

    ALIASES = LTL_SPEC_ALIASES
    BUCKET_DIR = LTL_SPEC_BUCKET_DIR
    WANDB_PROJECT = LTL_SPEC_WANDB_PROJECT

    def __init__(self, guarantees: list, assumptions: list = None, metadata: dict = None):
        self.guarantees = guarantees
        self.assumptions = assumptions if assumptions else []
        super().__init__(metadata=metadata)
        logger.info(
            "Successfully constructed dataset of %d guarantee patterns and %d assumption patterns",
            len(self.guarantees),
            len(self.assumptions),
        )

    def filter(
        self,
        max_ast_size: int,
        max_num_inputs: int,
        max_num_outputs: int,
        type: str = "guarantees",
    ) -> None:
        if type == "guarantees":
            props = self.guarantees
        elif type == "assumptions":
            props = self.assumptions
        else:
            raise Exception("Invalid type %s", type)

        counter = {"max_ast_size": 0, "num_inputs": 0, "num_outputs": 0}
        filtered_props = []
        for prop in props:
            tokens = lex_ltl(prop["pattern"])
            tokens = list(filter(lambda t: t not in ["(", ")"], tokens))
            num_inputs = len(prop["inputs"])
            num_outputs = len(prop["outputs"])
            invalid = False
            if max_ast_size and len(tokens) > max_ast_size:
                invalid = True
                counter["max_ast_size"] += 1
            if num_inputs > max_num_inputs:
                invalid = True
                counter["num_inputs"] += 1
            if num_outputs > max_num_outputs:
                invalid = True
                counter["num_outputs"] += 1
            if invalid:
                continue
            filtered_props.append(prop)

        if type == "assumptions":
            self.assumptions = filtered_props
        else:
            self.guarantees = filtered_props

        logger.info(
            "From filtered %s %d AST too large, %d too many inputs, %d too many outputs",
            type,
            counter["max_ast_size"],
            counter["num_inputs"],
            counter["num_outputs"],
        )

    def rename(
        self, inputs: list, outputs: list, shuffle_aps: bool = None, type: str = "guarantees"
    ) -> None:
        if type == "guarantees":
            props = self.guarantees
        elif type == "assumptions":
            props = self.assumptions
        else:
            raise Exception("Invalid type %s", type)

        renamed_props = []
        for prop in props:
            num_inputs = len(prop["inputs"])
            num_outputs = len(prop["outputs"])
            if shuffle_aps:
                renamed_inputs = list(random.choice(inputs, num_inputs, replace=False))
                renamed_outputs = list(random.choice(outputs, num_outputs, replace=False))
            else:
                renamed_inputs = inputs[:num_inputs]
                renamed_outputs = outputs[:num_outputs]
            renaming = dict(
                zip(prop["inputs"] + prop["outputs"], renamed_inputs + renamed_outputs)
            )
            prop_str = prop["pattern"]
            for ap, renamed_ap in renaming.items():
                # TODO doesn't work if ap is substring of other ap
                prop_str = prop_str.replace(ap, renamed_ap)
            # replace operators
            prop_str = prop_str.replace("&&", "&").replace("||", "|")
            sample = {"pattern": prop_str, "inputs": renamed_inputs, "outputs": renamed_outputs}
            renamed_props.append(sample)

        if type == "assumptions":
            self.assumptions = renamed_props
        else:
            self.guarantees = renamed_props

        logger.info("Renamed %s patterns", type)

    def save_to_path(self, path: str) -> None:
        assumptions_path = os.path.join(path, "assumptions.json")
        os.makedirs(os.path.dirname(assumptions_path), exist_ok=True)
        with open(assumptions_path, "w") as assumptions_file:
            json.dump({"patterns": self.assumptions}, assumptions_file, indent=2)
            logger.info(
                "Written %d assumption patterns to %s", len(self.assumptions), assumptions_path
            )

        guarantees_path = os.path.join(path, "guarantees.json")
        with open(guarantees_path, "w") as guarantees_file:
            json.dump({"patterns": self.guarantees}, guarantees_file, indent=2)
            logger.info(
                "Written %d guarantee patterns to %s", len(self.guarantees), guarantees_path
            )

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)
            logger.info("Written metadata to %s", metadata_path)

    @classmethod
    def load_from_path(cls, path: str):
        guarantees_path = os.path.join(path, "guarantees.json")
        with open(guarantees_path, "r") as guarantees_file:
            guarantees = json.load(guarantees_file)["patterns"]
            logger.info("Read in %d guarantee patterns", len(guarantees))

        assumptions_path = os.path.join(path, "assumptions.json")
        with open(assumptions_path, "r") as assumptions_file:
            assumptions = json.load(assumptions_file)["patterns"]
            logger.info("Read in %d assumption patterns", len(assumptions))

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
            logger.info("Read in metadata")

        return cls(guarantees, assumptions, metadata)

    @classmethod
    def from_ltl_spec_data(cls, data, unique_properties):
        assumptions = data.assumptions(unique=unique_properties)
        for assumption in assumptions:
            assumption["pattern"] = assumption.pop("assumption")
        guarantees = data.guarantees(unique=args.unique_properties)
        for guarantee in guarantees:
            guarantee["pattern"] = guarantee.pop("guarantee")
        return cls(guarantees, assumptions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extracts assumption and guarantee patterns from LTL specification data and writes them to a file"
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["i0", "i1", "i2", "i3", "i4"],
        help="list of input atomic propositions",
    )
    parser.add_argument(
        "--outputs",
        nargs="*",
        default=["o0", "o1", "o2", "o3", "o4"],
        help="list of output atomic propositions",
    )
    parser.add_argument(
        "--max-ast-size",
        type=int,
        default=None,
        help=("maximum size of the abstract syntax tree" "representing the pattern"),
    )
    parser.add_argument(
        "--shuffle-aps", action="store_true", help="shuffle atomic propositions when renaming"
    )
    parser.add_argument("--ltl-spec-data", default="sc-0", help="LTL specification data")
    parser.add_argument(
        "--unique-properties",
        action="store_true",
        help="assumptions and guarantees are checked for duplicates",
    )
    parser.add_argument(
        "--name", default="scp", help="name of the extracted LTL specification pattern dataset"
    )
    args = parser.parse_args()

    ltl_spec_data = LTLSpecData.load(args.ltl_spec_data)
    ltl_spec_pattern_data = LTLSpecPatternData.from_ltl_spec_data(
        ltl_spec_data, args.unique_properties
    )
    ltl_spec_pattern_data.filter(
        args.max_ast_size, len(args.inputs), len(args.outputs), "guarantees"
    )
    ltl_spec_pattern_data.filter(
        args.max_ast_size, len(args.inputs), len(args.outputs), "assumptions"
    )
    ltl_spec_pattern_data.rename(args.inputs, args.outputs, args.shuffle_aps, "guarantees")
    ltl_spec_pattern_data.rename(args.inputs, args.outputs, args.shuffle_aps, "assumptions")
    ltl_spec_pattern_data.metadata = {
        "ltl_spec_data": args.ltl_spec_data,
        "inputs": args.inputs,
        "outputs": args.outputs,
        "max_ast_size": args.max_ast_size,
        "shuffle_aps": args.shuffle_aps,
        "unique_properties": args.unique_properties,
    }
    ltl_spec_pattern_data.save(args.name, auto_version=True, upload=True)
