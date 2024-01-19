"""Script to run experiment"""

import argparse

from .experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML2 experiment")
    parser.add_argument("config_file")
    args = parser.parse_args()
    experiment = Experiment.from_config_file(args.config_file)
    experiment.run()
