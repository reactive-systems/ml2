"""Progress bars for progress actor"""

import json
from typing import Dict

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from ..datasets import DatasetWriter
from .progress_actor import ProgressActor


def progress_bar(
    progress_actor: ProgressActor, num_samples: int, stats_filepath: str = None
) -> None:
    pbar = tqdm(total=num_samples, desc="Generated samples", unit="sample")
    progress_actor.update.remote("samples", 0)
    while True:
        progress = ray.get(progress_actor.wait_for_update.remote())
        pbar.update(progress["samples"] - pbar.n)
        postfix_dict = dict(progress)
        postfix_dict.pop("samples", None)
        pbar.set_postfix(postfix_dict)
        if progress["samples"] >= num_samples:
            if stats_filepath:
                with open(stats_filepath, "w") as stats_file:
                    progress["elapsed"] = pbar.format_dict["elapsed"]
                    json.dump(progress, stats_file, indent=2)
            pbar.close()
            return


def data_writing_progress_bar(
    dataset_writer: DatasetWriter,
    progress_actor: ProgressActor,
    samples_queue: Queue,
    num_samples: int,
    stats_filepath: str = None,
) -> None:
    pbar = tqdm(total=num_samples, desc="Generated samples", unit="sample")
    progress_actor.update.remote("samples", 0)
    for _ in range(num_samples):
        sample = samples_queue.get(block=True)
        dataset_writer.add_sample(sample)
        progress = ray.get(progress_actor.get_progress.remote())
        pbar.update()
        postfix_dict = dict(progress)
        postfix_dict.pop("samples", None)
        pbar.set_postfix(postfix_dict)

    if stats_filepath:
        with open(stats_filepath, "w") as stats_file:
            progress["elapsed"] = pbar.format_dict["elapsed"]
            json.dump(progress, stats_file, indent=2)

    pbar.close()


def key_data_writing_progress_bar(
    dataset_writers: Dict[str, DatasetWriter],
    progress_actor: ProgressActor,
    samples_queue: Queue,
    num_samples: int,
    stats_filepath: str = None,
) -> None:
    pbar = tqdm(total=num_samples, desc="Generated samples", unit="sample")
    progress_actor.update.remote("samples", 0)
    for _ in range(num_samples):
        key, sample = samples_queue.get(block=True)
        if key not in dataset_writers:
            raise ValueError(f"Unexpected key {key}")
        dataset_writers[key].add_sample(sample)
        progress = ray.get(progress_actor.get_progress.remote())
        pbar.update()
        postfix_dict = dict(progress)
        postfix_dict.pop("samples", None)
        pbar.set_postfix(postfix_dict)

    if stats_filepath:
        with open(stats_filepath, "w") as stats_file:
            progress["elapsed"] = pbar.format_dict["elapsed"]
            json.dump(progress, stats_file, indent=2)

    pbar.close()
