"""LTL Synthesis using the Transformer"""

import csv
import json
import logging
import os
from statistics import median

from collections import Counter
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import wandb

from ml2.globals import WANDB_ENTITY

from ...aiger import AIGERSequenceEncoder, header_ints_from_str
from ...tools.nuxmv import nuXmv
from ...seq2seq_experiment import Seq2SeqExperiment
from ..ltl_spec import LTLSpecData
from .ltl_syn_data import LTLSynSplitData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLSynExperiment(Seq2SeqExperiment):

    BUCKET_DIR = "ltl-syn"
    WANDB_PROJECT = "ltl-syn"

    def __init__(
        self,
        aiger_order: list = None,
        aiger_unfold_negations: bool = False,
        aiger_unfold_latches: bool = False,
        batch_size: int = 256,
        dataset_name: str = "scpa-0",
        encode_realizable: bool = True,
        inputs: list = None,
        max_input_length: int = 128,
        max_target_length: int = 128,
        outputs: list = None,
        **kwargs,
    ):
        self.aiger_order = aiger_order if aiger_order else ["inputs", "latches", "outputs", "ands"]
        self.aiger_unfold_negations = aiger_unfold_negations
        self.aiger_unfold_latches = aiger_unfold_latches
        self.encode_realizable = encode_realizable
        self.inputs = inputs if inputs else ["i0", "i1", "i2", "i3", "i4"]
        self.outputs = outputs if outputs else ["o0", "o1", "o2", "o3", "o4"]
        super().__init__(
            batch_size=batch_size,
            dataset_name=dataset_name,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            **kwargs,
        )

    def analyze_eval_file(self, file: str):
        stats = {}
        with open(file, "r") as eval_file:
            csv_reader = csv.DictReader(eval_file)
            for row in csv_reader:
                target = row["target"]
                if not target:
                    continue
                _, _, num_latches, _, _ = header_ints_from_str(target.replace("\\n", "\n"))
                if num_latches not in stats:
                    stats[num_latches] = {}
                status = row["status"]
                stats[num_latches][status] = stats[num_latches].get(status, 0) + 1
        print(stats)

    def analyze_eval_circuits(
        self,
        directory: str,
        buckets: list = None,
        match: bool = False,
        property: str = "Max Var ID",
        title: str = None,
        width: float = 0.5,
        xlabel: str = None,
    ):
        stats = {}

        with open(os.path.join(directory, "log.csv"), "r") as eval_file:

            beam_status = {}

            for row in csv.DictReader(eval_file):

                status = row["status"]
                if status not in ["Match", "Satisfied", "Violated"]:
                    continue

                if row["beam"] == "0":
                    if beam_status:
                        for s in ["Match", "Satisfied", "Violated"]:
                            if s in beam_status:
                                value = median(beam_status[s])
                                if value not in stats:
                                    stats[value] = {}
                                stats[value][s] = stats[value].get(s, 0) + 1
                                break
                    beam_status = {}

                target = row["prediction"]
                (
                    max_var_id,
                    num_inputs,
                    num_latches,
                    num_outputs,
                    num_and_gates,
                ) = header_ints_from_str(target.replace("\\n", "\n"))

                if property == "Max Var ID":
                    value = max_var_id
                elif property == "Num Inputs":
                    value = num_inputs
                elif property == "Num Latches":
                    value = num_latches
                elif property == "Num Outputs":
                    value = num_outputs
                elif property == "Num AND Gates":
                    value = num_and_gates
                else:
                    raise ValueError(f"Unknown AIGER property {property}")

                if status not in beam_status:
                    beam_status[status] = []
                beam_status[status].append(value)

        filename = f'{property.lower().replace(" ", "-")}'

        with open(os.path.join(directory, f"{filename}.json"), "w") as f:
            json.dump(stats, f, indent=4, sort_keys=True)

        self.plot_eval_stats(
            stats,
            os.path.join(directory, f"{filename}.eps"),
            buckets=buckets,
            match=match,
            title=title,
            width=width,
            xlabel=xlabel,
        )

    def plot_eval_stats(
        self,
        stats,
        filepath: str,
        buckets: list = None,
        match: bool = False,
        title: str = None,
        width: float = 2.0,
        xlabel: str = None,
    ):

        keys = []
        matches = []
        satisfied = []
        violated = []

        for key in sorted(stats.keys()):
            keys.append(key)
            matches.append(stats[key].get("Match", 0))
            satisfied.append(stats[key].get("Satisfied", 0))
            violated.append(stats[key].get("Violated", 0))

        if buckets:
            bucket_matches, bucket_satisfied, bucket_violated = [], [], []
            i = 0
            for bucket in buckets:
                cum_matches, cum_satisfied, cum_violated = 0, 0, 0
                while i < len(keys) and keys[i] < bucket:
                    cum_matches += matches[i]
                    cum_satisfied += satisfied[i]
                    cum_violated += violated[i]
                    i += 1
                bucket_matches.append(cum_matches)
                bucket_satisfied.append(cum_satisfied)
                bucket_violated.append(cum_violated)
                logger.info("Bucket %d" % bucket)
                logger.info("Matches: %d" % cum_matches)
                logger.info("Satisfied: %d" % cum_satisfied)
                logger.info("Violated: %d" % cum_violated)
            keys = buckets
            matches = bucket_matches
            satisfied = bucket_satisfied
            violated = bucket_violated

        fig, ax = plt.subplots()
        match_or_satisfied = [sum(x) for x in zip(matches, satisfied)]
        if match:
            ax.bar(keys, matches, label="Match", color="tab:blue")
            ax.bar(keys, satisfied, width, bottom=matches, label="Satisfied", color="tab:green")
        else:
            ax.bar(keys, match_or_satisfied, width, label="Satisfied", color=(0, 0.5, 0, 0.7))
        p = ax.bar(
            keys,
            violated,
            width,
            bottom=match_or_satisfied,
            label="Violated",
            color=(0.5, 0.5, 0.5, 0.7),
        )
        perc = ["%.2f" % (x / (x + y)) for (x, y) in zip(match_or_satisfied, violated)]
        ax.bar_label(p, labels=perc)
        if buckets:
            ax.set_xticks(buckets)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.legend()
        plt.savefig(filepath, dpi=fig.dpi, facecolor="white", format="eps")

    @property
    def init_verifier(self):
        return nuXmv()

    def call(self, specification, training: bool = False, verify: bool = False):
        if not self.input_encoder.encode(specification):
            logger.info("Econding error: %s", self.input_encoder.error)
            return None
        formula_tensor, pos_enc_tensor = self.input_encoder.tensor
        # pylint: disable=E1102
        preds = self.eval_model(
            (tf.expand_dims(formula_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
            training=training,
        )[0]
        results = []
        for beam in preds[0]:
            if not self.target_encoder.decode(np.array(beam)):
                logger.info("Decoding error: %s", self.target_encoder.error)
                # return None
            beam_result = {}
            beam_result["circuit"] = self.target_encoder.circuit
            if verify:
                # pylint: disable=E1102
                beam_result["verification"] = self.verifier.model_check(
                    specification, beam_result["circuit"] + "\n"
                )
            results.append(beam_result)
        return results

    def wandb_log_stats(self, stats: Dict, beamsize: int, alpha: float, split: str):
        if self.stream_to_wandb:
            log_dict = {
                "data": split,
                "beam size": beamsize,
                "alpha": alpha,
                "samples": 0,
                "encoding_error": 0,
                "match": 0,
                "satisfied": 0,
                "decoding_error": 0,
                "violated": 0,
                "beam_search_satisfied": 0,
                "invalid": 0,
                "timeout": 0,
                "steps": 0,
                "error": 0,
                "accuracy": 0.0,
                "accuracy_encoded": 0.0,
            }

            for k, v in stats.items():
                log_dict[k] = v

            self.evaluation_table.add_data(*(list(log_dict.values())))

    def upload_wandb_log(self):
        index = self.evaluation_table.columns.index("accuracy")
        accuracy = 0.0
        for row in self.evaluation_table.data:
            accuracy = row[index] if row[index] > accuracy else accuracy
        wandb.run.log({"best_test_accuracy": accuracy})
        wandb.run.log({self.evaluation_table_name: self.evaluation_table})

    def eval_wandb_init(self):
        wandb.init(
            config=self.args,
            entity=WANDB_ENTITY,
            group=self.group,
            name=self.name,
            project=self.WANDB_PROJECT,
            id=self.wandb_run_id,
            resume="auto",
        )
        self.evaluation_table_name = "evaluation_metrics"
        self.evaluation_table = wandb.Table(
            columns=[
                "data",
                "beam size",
                "alpha",
                "samples",
                "encoding_error",
                "match",
                "satisfied",
                "decoding_error",
                "violated",
                "beam_search_satisfied",
                "invalid",
                "timeout",
                "steps",
                "error",
                "accuracy",
                "accuracy_enc",
            ]
        )

    def eval(
        self,
        alphas: list = None,
        beam_sizes: list = None,
        data: list = None,
        nuxmv_port: int = None,
        samples: int = 1024,
        shuffle_data: bool = True,
        training: bool = False,
    ):

        orig_batch_size = self.batch_size

        if not data:
            data = ["test", "syntcomp", "jarvis", "timeouts"]

        if nuxmv_port:
            self._verifier = nuXmv(port=nuxmv_port)

        if shuffle_data:
            self._dataset = None
            self.shuffle_on_load = True

        for alpha in alphas:
            self.alpha = alpha
            for beam_size in beam_sizes:
                self.beam_size = beam_size
                self.batch_size = orig_batch_size // beam_size
                if samples * beam_size < self.batch_size:
                    self.batch_size = samples
                steps = samples // self.batch_size

                if "test" in data:
                    logger.info(
                        "Evaluating testset for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_split(split="test", steps=steps, training=training, verify=True)

                if "syntcomp" in data:
                    logger.info(
                        "Evaluating SYNTCOMP 2020 with alpha %1.1f, beam_size %d and batch_size %d",
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_ltl_specs("sc-0", training=training)

                if "jarvis" in data:
                    logger.info(
                        "Evaluating smart home benchmarks with alpha %1.1f, beam_size %d and batch_size %d",
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_ltl_specs("jarvis-0", training=training)

                if "timeouts" in data:
                    logger.info(
                        "Evaluating timeouts for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_timeouts(steps=steps, training=training)

                self._eval_model = None

    def eval_ltl_specs(self, name: str, steps: int = None, training: bool = False):
        def spec_filter(spec):
            return spec.num_inputs <= len(self.inputs) and spec.num_outputs <= len(self.outputs)

        LTLSpecData.download(name)
        spec_ds = LTLSpecData.from_bosy_files(LTLSpecData.local_path(name), spec_filter)
        spec_ds.rename_aps(self.inputs, self.outputs)
        for spec in spec_ds.dataset:
            spec.inputs = self.inputs
            spec.outputs = self.outputs

        self.eval_generator(
            spec_ds.dataset,
            name,
            includes_target=False,
            steps=steps,
            training=training,
            verify=True,
        )

    def eval_timeouts(self, steps: int = None, training: bool = False):
        timeouts = self.dataset["timeouts"]
        timeouts = [sample for sample, _ in timeouts.generator()]
        self.eval_generator(
            timeouts,
            "timeouts",
            includes_target=False,
            steps=steps,
            training=training,
            verify=True,
        )

    def eval_generator(
        self,
        generator,
        name: str,
        includes_target: bool = False,
        steps: int = None,
        training: bool = False,
        verify: bool = False,
    ):
        folder = f"a{self.alpha}-bs{self.beam_size}"
        if steps:
            folder += f"-n{steps * self.batch_size}"
        eval_gen_dir = os.path.join(os.path.join(self.eval_dir, name), folder)
        if not os.path.isdir(eval_gen_dir):
            os.makedirs(eval_gen_dir)

        log_filepath = os.path.join(eval_gen_dir, "log.csv")
        log_file = open(log_filepath, "w")
        fieldnames = ["beam", "status", "problem", "prediction", "target"]
        file_writer = csv.DictWriter(log_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        file_writer.writeheader()
        counters = Counter()
        pbar = tqdm(desc="Evaluated samples", unit="sample")
        problem_batch, formula_batch, pos_enc_batch, target_batch = [], [], [], []

        def eval_batch():
            nonlocal counters, problem_batch, formula_batch, pos_enc_batch, target_batch, pbar
            batch_dataset = tf.data.Dataset.from_tensor_slices((formula_batch, pos_enc_batch))
            batch = next(iter(batch_dataset.batch(self.batch_size, drop_remainder=False)))
            predictions, _ = self.eval_model(batch, training=training)  # pylint: disable=E1102
            for i, pred in enumerate(predictions):
                any_beam_satisfied = False
                problem = problem_batch[i]
                target = target_batch[i] if includes_target else ""
                problem_name = problem.name if problem.name else problem.formula_str
                for beam_id, beam in enumerate(pred):
                    row = {
                        "beam": beam_id,
                        "problem": problem_name,
                        "prediction": "",
                        "target": target.replace("\n", "\\n"),
                    }
                    if not self.target_encoder.decode(np.array(beam)):
                        row["status"] = f"Decoding Error {self.target_encoder.error}"
                        row["prediction"] = np.array2string(
                            np.array(beam), max_line_width=3 * self.max_target_length
                        )
                        file_writer.writerow(row)
                        counters["decoding_error"] += 1
                        continue
                    realizable = self.target_encoder.realizable  # True # 'i0 i0' in target
                    circuit = self.target_encoder.circuit
                    row["prediction"] = circuit.replace("\n", "\\n")
                    if includes_target:
                        if circuit == target:
                            row["status"] = "Match"
                            counters["match"] += 1
                    # pylint: disable=E1102
                    result = self.verifier.model_check(problem, circuit + "\n", realizable)
                    counters[result.value] += 1
                    if result.value == "satisfied":
                        any_beam_satisfied = True
                    if "status" not in row:
                        row["status"] = result.value.capitalize()
                    else:
                        if row["status"] == "Match" and result.value != "satisfied":
                            logger.warning("Match not satisfied")
                    file_writer.writerow(row)
                if any_beam_satisfied:
                    counters["beam_search_satisfied"] += 1
                pbar.update()
                pbar.set_postfix(counters)
            problem_batch, formula_batch, pos_enc_batch, target_batch = [], [], [], []
            counters["steps"] += 1

        for sample in generator:
            counters["samples"] += 1
            problem = sample[0] if includes_target else sample
            problem_name = problem.name if problem.name else problem.formula_str
            target = sample[1] if includes_target else None
            row = {
                "beam": 0,
                "problem": problem_name,
                "prediction": "",
                "target": target.replace("\n", "\\n") if target else "",
            }
            if not self.input_encoder.encode(problem):
                row["status"] = f"Encoding Error {self.input_encoder.error}"
                file_writer.writerow(row)
                counters["encoding_error"] += 1
                pbar.update()
            elif includes_target and not self.target_encoder.encode(target):
                row["status"] = f"Target Error {self.target_encoder.error}"
                file_writer.writerow(row)
                counters["target_error"] += 1
                pbar.update()
            else:
                problem_batch.append(problem)
                formula_tensor, pos_enc_tensor = self.input_encoder.tensor
                formula_batch.append(formula_tensor)
                pos_enc_batch.append(pos_enc_tensor)
                if includes_target:
                    target_batch.append(sample[1])

            if counters["samples"] % self.batch_size == 0 and problem_batch:
                eval_batch()
                if steps and counters["steps"] >= steps:
                    break

        if problem_batch:
            eval_batch()

        pbar.close()
        log_file.close()
        stats = counters
        stats["accuracy"] = counters["beam_search_satisfied"] / counters["samples"]
        stats["accuracy_encoded"] = counters["beam_search_satisfied"] / (
            counters["samples"] - counters["encoding_error"]
        )
        stats_filepath = os.path.join(eval_gen_dir, "stats.json")
        with open(stats_filepath, "w") as stats_file:
            json.dump(stats, stats_file, indent=4)
        self.wandb_log_stats(stats=stats, beamsize=self.beam_size, alpha=self.alpha, split=name)

    @property
    def eval_paths(self) -> list:
        def subfolders(path):
            return [e.path for e in os.scandir(path) if e.is_dir() and not e.name.startswith(".")]

        return [p2 for p1 in subfolders(self.eval_dir) for p2 in subfolders(p1)]

    @property
    def init_dataset(self):
        return LTLSynSplitData.load(self.dataset_name)

    @property
    def init_target_encoder(self):
        return AIGERSequenceEncoder(
            start=True,
            eos=True,
            pad=self.max_target_length,
            components=self.aiger_order,
            encode_start=False,
            encode_realizable=self.encode_realizable,
            inputs=self.inputs,
            outputs=self.outputs,
            unfold_negations=self.aiger_unfold_negations,
            unfold_latches=self.aiger_unfold_latches,
        )

    @classmethod
    def analyze_eval_file_realizability(cls, file: str):
        stats = {
            "Realizable": {
                "Beam Search Satisfied": 0,
                "Samples": 0,
                "Correct Realizability Prediction": 0,
            },
            "Unrealizable": {
                "Beam Search Satisfied": 0,
                "Samples": 0,
                "Correct Realizability Prediction": 0,
            },
        }
        with open(file, "r") as eval_file:
            csv_reader = csv.DictReader(eval_file)
            beam_satisfied = False
            beam_realizable = 0
            beam_unrealizable = 0
            realizable = None
            # last_realizable = None
            for row in csv_reader:
                if row["beam"] == "0" and realizable:
                    # if last_realizable:
                    #     beam_realizable -= 1
                    # else:
                    #     beam_unrealizable -= 1
                    if (realizable == "Realizable" and beam_realizable > beam_unrealizable) or (
                        realizable == "Unrealizable" and beam_unrealizable > beam_realizable
                    ):
                        stats[realizable]["Correct Realizability Prediction"] += 1
                    if beam_realizable == beam_unrealizable:
                        if "Tie" not in stats[realizable]:
                            stats[realizable]["Tie"] = 0
                        stats[realizable]["Tie"] += 1
                    beam_realizable = 0
                    beam_unrealizable = 0
                realizable = "Realizable" if "i0 i0" in row["target"] else "Unrealizable"
                status = row["status"]
                # if realizable == 'Realizable' and 'i0 i0' in row['prediction']:
                #     stats[realizable]['Correct Realizability Prediction'] += 1
                # if realizable == 'Unrealizable' and 'i0 o0' in row['prediction']:
                #     stats[realizable]['Correct Realizability Prediction'] += 1
                if "i0 i0" in row["prediction"]:
                    beam_realizable += 1
                    # last_realizable = True
                if "i0 o0" in row["prediction"]:
                    beam_unrealizable += 1
                    # last_realizable = False
                if status.startswith("Decoding Error"):
                    status = "Decoding Error"
                if row["beam"] == "0":
                    beam_satisfied = False
                    stats[realizable]["Samples"] += 1
                if status == "Satisfied" or status == "Match":
                    if not beam_satisfied:
                        beam_satisfied = True
                        stats[realizable]["Beam Search Satisfied"] += 1
                if status not in stats[realizable]:
                    stats[realizable][status] = 0
                stats[realizable][status] = stats[realizable][status] + 1
            if (realizable == "Realizable" and beam_realizable > beam_unrealizable) or (
                realizable == "Unrealizable" and beam_unrealizable > beam_realizable
            ):
                stats[realizable]["Correct Realizability Prediction"] += 1
        return stats

    @classmethod
    def add_eval_args(cls, parser):
        super().add_eval_args(parser)
        parser.add_argument("-d", "--data", nargs="*", default=None)
        parser.add_argument("--nuxmv-port", type=int, default=None)

    @classmethod
    def add_init_args(cls, parser):
        super().add_init_args(parser)
        defaults = cls.get_default_args()
        parser.add_argument("--aiger-order", nargs="*", default=defaults["aiger_order"])
        parser.add_argument("--aiger-unfold-negations", action="store_true")
        parser.add_argument("--aiger-unfold-latches", action="store_true")
        parser.add_argument(
            "--no-encode-realizable", action="store_false", dest="encode_realizable"
        )
        parser.add_argument("--inputs", nargs="*", default=defaults["inputs"])
        parser.add_argument("--outputs", nargs="*", default=defaults["outputs"])
