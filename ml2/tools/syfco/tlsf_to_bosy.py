"""Translation of TLSF specifications to BoSy input format"""

import argparse
import os

from .syfco import Syfco


def tlsf_to_bosy(
    tlsf_dir,
    bosy_dir=None,
    semantics: str = None,
):
    syfco = Syfco()

    if bosy_dir is None:
        bosy_dir = tlsf_dir

    success_counter = 0
    failure_counter = 0

    for file in os.listdir(tlsf_dir):
        if file.endswith(".tlsf"):
            filename, _ = os.path.splitext(file)
            filepath_tlsf = os.path.join(tlsf_dir, file)
            filepath_bosy = os.path.abspath(os.path.join(bosy_dir, filename + ".json"))

            spec = syfco.from_tlsf_file(filepath_tlsf)
            spec.semantics = semantics
            spec.to_bosy_file(filepath_bosy)

            print(f"Successfully converted {filepath_tlsf} to {filepath_bosy}")
            # format json file
            print(f"Successfully formatted {filepath_bosy}")
            success_counter += 1

    print(f"Failed to convert {failure_counter} tlsf files")
    print(f"Successfully converted and formatted {success_counter} tlsf files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts all TLSF files in the specified directory to JSON files that are conform with the BoSy input format using SyFCo"
    )
    parser.add_argument("--dir-tlsf", type=str, default=None, help="Directory with TLSF files")
    parser.add_argument(
        "--dir-bosy", type=str, default=None, help="Directory where BoSy input files are written"
    )
    parser.add_argument("--semantics", type=str, default=None, help="Mealy or Moore")
    args = parser.parse_args()
    if args.dir_tlsf is not None:
        tlsf_to_bosy(args.dir_tlsf, args.dir_bosy, args.semantics)
    else:
        print("Please specify a directory")
