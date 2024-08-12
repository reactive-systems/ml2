"""Converts a dataset of resolution proofs to a dataset of binary resolution proofs, i.e., each resolvent has exactly two antecedents"""

import argparse

from tqdm import tqdm

from ...datasets import CSVDataset, SplitDataset, load_dataset
from ...tools.booleforce import BooleForce
from .cnf_res_problem import CNFResProblem


def main(args):
    dataset = load_dataset(name=args.input_dataset, project=args.input_project)

    new_dataset = SplitDataset(
        dtype=CNFResProblem, name=args.output_dataset, project=args.output_project
    )

    booleforce = BooleForce()

    for split_name in dataset.split_names:
        new_dataset_split = CSVDataset(
            name=f"{args.output_dataset}/{split_name}",
            dtype=CNFResProblem,
            project=args.output_project,
        )
        split = dataset[split_name]
        for sample in tqdm(split.generator(), desc=split_name + " split", total=split.size):
            sample.proof = booleforce.binarize_res_proof(sample.proof)
            new_dataset_split.add_sample(sample)
        new_dataset[split_name] = new_dataset_split

    new_dataset.save(recurse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to binarize dataset of resolution proofs",
        epilog="Example usage: --input-dataset res-2 --output-dataset res-4",
    )
    parser.add_argument("--input-dataset", type=str, required=True, help="Dataset to binarize")
    parser.add_argument("--input-project", type=str, default="prop-res", help="Input project")
    parser.add_argument("--output-dataset", type=str, required=True, help="New binary dataset")
    parser.add_argument("--output-project", type=str, default="prop-res", help="Output project")
    main(parser.parse_args())
