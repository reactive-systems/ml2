"""Induces a probabilistic context-free grammar from a set of LTL specification patterns"""

import argparse
import logging
import os
from typing import List

from nltk import CFG
from nltk.grammar import PCFG, induce_pcfg
from nltk.parse import BottomUpChartParser
from tqdm import tqdm

from ....datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INPUT_NT = "INP"
OUTPUT_NT = "OUT"
TRUE_NT = "TRUE"
FALSE_NT = "FALSE"
UN_OP_TO_NT = {"!": "N", "X": "X", "F": "F", "G": "G"}
BIN_OP_TO_NT = {"&": "A", "|": "O", "->": "I", "<->": "E", "U": "U", "R": "R"}


def induce_pcfg_from_cfg(
    cfg_path: str,
    dataset: str,
    max_num_inputs: int = 10,
    max_num_outputs: int = 10,
    max_prop_size: int = 25,
) -> PCFG:
    with open(cfg_path, "r") as cfg_file:
        cfg = CFG.fromstring("\n".join(cfg_file.readlines()[1:]))
    parser = BottomUpChartParser(cfg)

    ds = load_dataset(dataset)

    productions = []
    for prop in tqdm(ds.generator()):
        if (
            prop.num_inputs > max_num_inputs
            or prop.num_outputs > max_num_outputs
            or prop.size() > max_prop_size
        ):
            continue
        tokens = prop.to_tokens(notation="prefix")
        result = parser.parse(tokens)
        for tree in result:
            productions += tree.productions()

    logger.info(f"Derived {len(productions)} productions")

    pcfg = induce_pcfg(cfg.start(), productions)

    return pcfg


def construct_productions(
    inputs: List[str],
    outputs: List[str],
    unary_ops: List[str],
    binary_ops: List[str],
) -> List[str]:
    productions = []

    un_nts = [nt for op, nt in UN_OP_TO_NT.items() if op in unary_ops]
    bin_nts = [nt for op, nt in BIN_OP_TO_NT.items() if op in binary_ops]
    atomic_nts = bin_nts + un_nts + [INPUT_NT, OUTPUT_NT, TRUE_NT, FALSE_NT]

    for operator in un_nts:
        for operand in atomic_nts:
            productions.append(f"S -> {operator}OP {operand}")
            productions.append(f"{operator} -> {operator}OP {operand}")

    for operand1 in atomic_nts:
        for operand2 in atomic_nts:
            productions.append(f"{operand1 + operand2} -> {operand1} {operand2}")
            for operator in bin_nts:
                productions.append(f"S -> {operator}OP {operand1 + operand2}")
                productions.append(f"{operator} -> {operator}OP {operand1 + operand2}")

    for op in unary_ops:
        productions.append(f'{UN_OP_TO_NT[op]}OP -> "{op}"')

    for op in binary_ops:
        productions.append(f'{BIN_OP_TO_NT[op]}OP -> "{op}"')

    for i in inputs:
        productions.append(f'{INPUT_NT} -> "{i}"')

    for o in outputs:
        productions.append(f'{OUTPUT_NT} -> "{o}"')

    productions.append(f'{TRUE_NT} -> "true"')
    productions.append(f'{FALSE_NT} -> "false"')

    return productions


def main(path: str, dataset: str):
    productions = construct_productions(
        inputs=["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"],
        outputs=["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"],
        unary_ops=["!", "X", "F", "G"],
        binary_ops=["&", "|", "->", "<->", "U", "R"],
    )
    logger.info(f"Constructed {len(productions)} productions")

    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Constructed path {path}")

    cfg = CFG.fromstring("\n".join(productions))
    cfg_path = os.path.join(path, "cfg.txt")
    with open(cfg_path, "w") as cfg_file:
        cfg_file.write(str(cfg))
    logger.info(f"Written CFG to {cfg_path}")

    pcfg = induce_pcfg_from_cfg(cfg_path=cfg_path, dataset=dataset)

    pcfg_path = os.path.join(path, "pcfg.txt")
    with open(pcfg_path, "w") as pcfg_file:
        pcfg_file.write(str(pcfg))
    logger.info(f"Written PCFG to {pcfg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds a probabilistic context-free grammar")
    parser.add_argument("--path", type=str, required=True, help="path to save PCFG")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="ltl-spec/scp-1",
        help="datset to induce PCFG",
    )
    args = parser.parse_args()
    main(path=args.path, dataset=args.dataset)
