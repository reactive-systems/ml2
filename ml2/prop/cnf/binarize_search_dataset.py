"""Converts the resolution proofs in a SAT search dataset to binary resolution proofs, i.e., each resolvent has exactly two antecedents"""

import argparse

from tqdm import tqdm

from ...datasets import CSVDataset, SplitDataset, load_dataset
from ...tools.booleforce import BooleForce
from .cnf_sat_search_problem import CNFSatSearchProblem


def main(args):
    dataset = load_dataset(name=args.input_dataset, project=args.input_project)

    new_dataset = SplitDataset(
        dtype=CNFSatSearchProblem, name=args.output_dataset, project=args.output_project
    )

    booleforce = BooleForce()

    for split_name in dataset.split_names:
        split = dataset[split_name]
        new_dataset_split = CSVDataset(
            name=f"{args.output_dataset}/{split_name}",
            dtype=CNFSatSearchProblem,
            project=args.output_project,
            filename=split.filename,
            sep=split.sep,
        )
        for sample in tqdm(split.generator(), desc=split_name + " split", total=split.size):
            if sample.solution.res_proof is not None:
                sample.solution.res_proof = booleforce.binarize_res_proof(
                    sample.solution.res_proof
                )
            new_dataset_split.add_sample(sample)
        new_dataset[split_name] = new_dataset_split

    new_dataset.save(recurse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to binarize resolution proofs in a dataset of SAT search problems",
        epilog="Example usage: --input-dataset rcagr-0 --output-dataset rcar-0",
    )
    parser.add_argument("--input-dataset", type=str, required=True, help="Dataset to binarize")
    parser.add_argument("--input-project", type=str, default="prop-sat", help="Input project")
    parser.add_argument("--output-dataset", type=str, required=True, help="New binarized dataset")
    parser.add_argument("--output-project", type=str, default="prop-sat", help="Output project")
    main(parser.parse_args())
