"""Prototypical dataset classes"""

import csv
from enum import Enum
import json
import logging
import os.path
import random
import tensorflow as tf
from typing import Dict

from ..artifact import Artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Data(Artifact):

    WANDB_TYPE = "data"

    def __init__(self, data_frame=None, metadata: dict = None):
        self.data_frame = data_frame
        super().__init__(metadata=metadata)

    def generator(self):
        """Yields data samples"""
        raise NotImplementedError

    def shuffle(self):
        """Shuffles the data"""
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)

    @property
    def size(self) -> int:
        """Number of the data samples"""
        return len(self.data_frame.index)

    def stats(self):
        """Statistics of the data"""
        raise NotImplementedError

    def tf_generator(self):
        """Yields tensors representing encoded samples"""
        raise NotImplementedError

    @classmethod
    def stats_path(cls, name: str) -> str:
        return os.path.join(cls.local_path(name), "stats")


class SplitData(Artifact):

    WANDB_TYPE = "split_data"

    def __init__(self, splits: dict = None, metadata: dict = None):
        self._splits = splits if splits else {}
        super().__init__(metadata=metadata)

    def __getitem__(self, name):
        if name not in self.split_names:
            raise ValueError(
                (
                    f"Split name {name} does not match any of "
                    f"the available split names {self.split_names}"
                )
            )
        return self._splits[name]

    def __setitem__(self, name, split):
        self._splits[name] = split

    def generator(self, splits=None):
        """Yields dataset samples"""
        split_names = splits if splits else self.split_names
        for name in split_names:
            split = self._splits[name]
            for sample in split.generator():
                yield sample

    def shuffle(self, splits=None):
        split_names = splits if splits else self.split_names
        for name in split_names:
            self._splits[name].shuffle()

    @property
    def size(self):
        """Size of each split"""
        return {name: split.size for name, split in self._splits.items()}

    @property
    def split_names(self):
        return [*self._splits]

    def stats(self):
        """Statistics of each split"""
        return {name: split.stats() for name, split in self._splits.items()}

    def save_to_path(self, path: str) -> None:
        """Saves splits to a folder"""
        for name, split in self._splits.items():
            split_path = os.path.join(path, name)
            split.save_to_path(split_path)
            logger.info("Written %s data to %s", name, split_path)

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)
            logger.info("Written metadata to %s", metadata_path)

    @classmethod
    def stats_path(cls, name: str) -> str:
        return os.path.join(cls.local_path(name), "stats")


class UnsupervisedData(Data):
    def __init__(self, data_frame):
        self.encoder = None
        self.encoder_errors = {}
        super().__init__(data_frame)

    def tf_dataset(self, encoder):
        self.encoder = encoder
        output_types = encoder.tensors_dtype
        return tf.data.Dataset.from_generator(self.tf_generator, output_types)

    def tf_generator(self):
        for sample in self.generator():
            if not self.encoder.encode(sample):
                error = self.encoder.error
                self.encoder_errors[error] = self.encoder_errors.get(error, 0) + 1
                continue
            encoding = self.encoder.tensors
            yield encoding


class SplitUnsupervisedData(SplitData):
    @property
    def encoder_errors(self):
        return {name: split.encoder_errors for name, split in self._splits.items()}

    def tf_dataset(self, encoder, splits=None):
        """Constructs for each split a tensorflow dataset given an encoder"""
        split_names = splits if splits else self.split_names
        return {
            name: split.tf_dataset(encoder)
            for name, split in self._splits.items()
            if name in split_names
        }


class SupervisedData(Data):
    def __init__(self, data_frame, metadata: dict = None):
        self.input_encoder = None
        self.target_encoder = None
        self.input_encoder_errors: Dict[str, int] = {}
        self.target_encoder_errors: Dict[str, int] = {}
        super().__init__(data_frame, metadata)

    def input_generator(self):
        for inp, _ in self.generator():
            yield inp

    def target_generator(self):
        for _, tar in self.generator():
            yield tar

    def tf_dataset(self, input_encoder, target_encoder):
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.input_encoder_errors = {}
        self.target_encoder_errors = {}
        output_signature = (input_encoder.tensor_spec, target_encoder.tensor_spec)
        return tf.data.Dataset.from_generator(self.tf_generator, output_signature=output_signature)

    def tf_generator(self):
        for inp, tar in self.generator():
            if not self.input_encoder.encode(inp):
                error = self.input_encoder.error
                self.input_encoder_errors[error] = self.input_encoder_errors.get(error, 0) + 1
                continue
            input_tensor = self.input_encoder.tensor
            if not self.target_encoder.encode(tar):
                error = self.target_encoder.error
                self.target_encoder_errors[error] = self.target_encoder_errors.get(error, 0) + 1
                continue
            target_tensor = self.target_encoder.tensor
            yield input_tensor, target_tensor


class SplitSupervisedData(SplitData):
    def input_generator(self, splits=None):
        split_names = splits if splits else self.split_names
        for name in split_names:
            split = self._splits[name]
            for sample in split.input_generator():
                yield sample

    def target_generator(self, splits=None):
        split_names = splits if splits else self.split_names
        for name in split_names:
            split = self._splits[name]
            for sample in split.target_generator():
                yield sample

    @property
    def input_encoder_errors(self):
        return {name: split.input_encoder_errors for name, split in self._splits.items()}

    @property
    def target_encoder_errors(self):
        return {name: split.target_encoder_errors for name, split in self._splits.items()}

    def tf_dataset(self, input_encoder, target_encoder, splits=None):
        """Constructs for each split a tensorflow dataset given an input encoder
        and a target encoder"""
        split_names = splits if splits else self.split_names
        return {
            name: split.tf_dataset(input_encoder, target_encoder)
            for name, split in self._splits.items()
            if name in split_names
        }


def from_csv_file(filepath, row_to_sample):
    data = []
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row_to_sample(row))
    return data


def from_csv_files(csv_files_dir, row_to_sample):
    data = {}
    for split in Split:
        filepath = os.path.join(csv_files_dir, split.value + ".csv")
        if os.path.isfile(filepath):
            data[split] = from_csv_file(filepath, row_to_sample)
    return data


def to_csv_files(csv_files_dir, data, sample_to_row):
    for split in Split:
        filepath = os.path.join(csv_files_dir, split.value + ".csv")
        with open(filepath, "w") as split_file:
            writer = csv.writer(split_file, quoting=csv.QUOTE_ALL)
            for sample in data[split]:
                row = sample_to_row(sample)
                writer.writerow(row)


def from_json_files(json_files_dir):
    data = {}
    for split in Split:
        filepath = os.path.join(json_files_dir, split.value + ".json")
        if os.path.isfile(filepath):
            with open(filepath, "r") as split_file:
                split_data = json.load(split_file)
                data[split] = split_data["dataset"]
    return data


def split_and_save_to_json(json_files_dir, samples, train_frac, val_frac, shuffle=True):
    if shuffle:
        random.Random().shuffle(samples)
    num_samples = len(samples)
    filepath = os.path.join(dir, "samples.json")
    with open(filepath, "w") as file:
        json.dump({"dataset": samples}, file, indent=2)
    splits = {}
    splits[Split.TRAIN] = samples[0 : int(train_frac * num_samples)]
    splits[Split.VAL] = samples[
        int(train_frac * num_samples) : int((train_frac + val_frac) * num_samples)
    ]
    splits[Split.TEST] = samples[int((train_frac + val_frac) * num_samples) :]
    for split in Split:
        filepath = os.path.join(json_files_dir, split.value + ".json")
        with open(filepath, "w") as file:
            split_dict = {"dataset": splits[split]}
            json.dump(split_dict, file, indent=2)
            logging.info("%d %s samples written to %s", len(splits[split]), split.value, filepath)


def add_data_gen_args(parser):
    parser.add_argument("--train-frac", type=float, default=0.8, metavar="fraction")
    parser.add_argument("--val-frac", type=float, default=0.1, metavar="fraction")
    parser.add_argument("--test-frac", type=float, default=0.1, metavar="fraction")
