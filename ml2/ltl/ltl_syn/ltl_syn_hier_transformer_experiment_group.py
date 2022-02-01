import json
import logging
import os

import numpy as np

from ...artifact import Artifact
from .ltl_syn_hier_transformer_experiment import LTLSynHierTransformerExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSynHierTransformerExperimentGroup(Artifact):

    BUCKET_DIR = "ltl-syn"
    WANDB_PROJECT = "ltl-syn"
    WANDB_TYPE = "experiment-group"

    def __init__(self, experiments: list, name: str = "hier-transformer-group"):
        self.name = name
        self.experiments = experiments
        logger.info("Created experiment group containing %d experiments" % len(self.experiments))
        metadata = {"experiments": [experiment.name for experiment in self.experiments]}
        super().__init__(name=name, metadata=metadata)

    def avg_eval(self):
        ref_experiment = self.experiments[0]
        for p in ref_experiment.eval_paths:
            eval_name = p.replace(ref_experiment.eval_dir + "/", "")
            logger.info("Averaging evaluation %s" % eval_name)

            stats = []
            for experiment in self.experiments:
                stats_filepath = os.path.join(
                    os.path.join(experiment.eval_dir, eval_name), "stats.json"
                )
                with open(stats_filepath, "r") as stats_file:
                    stats.append(json.load(stats_file))

            avg_stats = {}
            std_dev_stats = {}
            for k in stats[0]:
                k_stats = [s[k] if k in s else 0 for s in stats]
                avg_stats[k] = np.mean(k_stats)
                std_dev_stats[k] = np.std(k_stats)

            eval_dir = os.path.join(os.path.join(self.local_path(self.name), eval_name))
            if not os.path.isdir(eval_dir):
                os.makedirs(eval_dir)

            avg_eval_filepath = os.path.join(eval_dir, "avg_stats.json")
            with open(avg_eval_filepath, "w") as avg_eval_file:
                json.dump(avg_stats, avg_eval_file, indent=4)

            std_eval_filepath = os.path.join(eval_dir, "std_stats.json")
            with open(std_eval_filepath, "w") as std_eval_file:
                json.dump(std_dev_stats, std_eval_file, indent=4)

    def avg_realizability(self):
        ref_experiment = self.experiments[0]
        for p in ref_experiment.eval_paths:
            eval_name = p.replace(ref_experiment.eval_dir + "/", "")
            print(f"Averaging realizability for evaluation {eval_name}")

            real_acc = []
            for experiment in self.experiments:
                log_file = os.path.join(os.path.join(experiment.eval_dir, eval_name), "log.csv")
                stats = LTLSynHierTransformerExperiment.analyze_eval_file_realizability(log_file)
                true_real_pred = (
                    stats["Realizable"]["Correct Realizability Prediction"]
                    + stats["Unrealizable"]["Correct Realizability Prediction"]
                )
                samples = stats["Realizable"]["Samples"] + stats["Unrealizable"]["Samples"]
                encoding_errors = 0
                for r in ["Realizable", "Unrealizable"]:
                    for k in stats[r].keys():
                        if k.startswith("Encoding Error"):
                            encoding_errors += stats[r][k]
                real_acc.append(true_real_pred / (samples - encoding_errors))

            print(
                {
                    "Realizability Acc Mean": np.mean(real_acc),
                    "Realizability Acc Std Dev": np.std(real_acc),
                }
            )

    @classmethod
    def create(cls, name: str, experiments: list):
        loaded_exps = []
        for e in experiments:
            loaded_exps.append(LTLSynHierTransformerExperiment.load(e))
        return cls(loaded_exps, name)
