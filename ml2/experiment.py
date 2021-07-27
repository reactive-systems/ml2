"""Base experiment class"""

import argparse
import inspect
import json
import logging
import os
import sys
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf

from .artifact import Artifact
from .gcp_bucket import latest_version
from .globals import WANDB_ENTITY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dtype_float_str_to_class = {
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64
}
dtype_float_class_to_str = {c: s for s, c in dtype_float_str_to_class.items()}

dtype_int_str_to_class = {
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64
}
dtype_int_class_to_str = {c: s for s, c in dtype_int_str_to_class.items()}


class Experiment(Artifact):

    WANDB_TYPE = 'experiment'

    def __init__(self,
                 auto_version: bool = True,
                 batch_size: int = 32,
                 cache_dataset: bool = True,
                 checkpoint_monitor: str = 'val_loss',
                 dataset_name: str = None,
                 drop_batch_remainder: bool = True,
                 dtype_float: str = 'float32',
                 dtype_int: str = 'int32',
                 initial_steps: int = 0,
                 name: str = None,
                 parent_name: str = None,
                 steps: int = 100,
                 shuffle_on_load: bool = True,
                 tf_shuffle_buffer_size: int = 0,
                 val_freq: int = 10):
        self.batch_size = batch_size
        self.cache_dataset = cache_dataset
        self.checkpoint_monitor = checkpoint_monitor
        self.dataset_name = dataset_name
        self.drop_batch_remainder = drop_batch_remainder
        if dtype_float not in dtype_float_str_to_class:
            sys.exit(f'Unrecognized float data type argument {dtype_float}')
        self.dtype_float = dtype_float_str_to_class[dtype_float]
        if dtype_int not in dtype_int_str_to_class:
            sys.exit(f'Unrecognized integer data type argument {dtype_int}')
        self.dtype_int = dtype_int_str_to_class[dtype_int]
        if auto_version:
            version = latest_version(self.BUCKET_DIR, name) + 1
            name += f'-{version}'
        self.name = name
        self.parent_name = parent_name if parent_name else ''
        self.shuffle_on_load = shuffle_on_load
        self.steps = steps
        self.tf_shuffle_buffer_size = tf_shuffle_buffer_size
        self.val_freq = val_freq

        if self.parent_name and not initial_steps:
            parent_experiment = self.load(self.parent_name)
            self.initial_steps = parent_experiment.steps  # // self.validation_freq
        elif initial_steps:
            self.initial_steps = initial_steps
        else:
            self.initial_steps = 0

        self._dataset = None
        self._eval_model = None
        self._learning_rate = None
        self._optimizer = None
        self._prepared_tf_dataset = {}
        self._solver = None
        self._tf_dataset = {}
        self._train_model = None
        self._verifier = None

        logger.info('Initialized experiment with arguments:\n%s',
                    '\n'.join([f'{a}: {v}' for a, v in self.args.items()]))

        if os.path.exists(self.local_dir):
            logger.warning('Experiment %s already exists locally', self.name)
        else:
            #create model directory
            os.makedirs(self.local_dir)
            logger.info('Created experiment directory %s', self.local_dir)

        super().__init__(metadata=self.args)

    def save_to_path(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info('Created directory %s', path)
        args_filepath = os.path.join(path, 'args.json')
        with open(args_filepath, 'w') as args_file:
            json.dump(self.str_args, args_file, indent=2)
        logger.info('Written experiment arguments to %s', args_filepath)

    @property
    def args(self):
        return {
            k: v
            for k, v in sorted(self.__dict__.items())
            if not k.startswith('_') and k != 'metadata'
        }

    @property
    def callbacks(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path(self.name),
            monitor=self.checkpoint_monitor,
            save_weights_only=True,
            save_best_only=True,
            verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.local_path(self.name))
        return [checkpoint_callback, tensorboard_callback]

    @property
    def dataset(self):
        if self._dataset:
            return self._dataset
        self._dataset = self.init_dataset
        if self.shuffle_on_load:
            self._dataset.shuffle()
            logger.info('Shuffled dataset')
        return self._dataset

    def eval(self):
        raise NotImplementedError()

    def eval_split(self,
                   split: str,
                   steps: int = None,
                   training: bool = False,
                   verify: bool = False):
        generator = self.dataset.generator(splits=[split])
        self.eval_generator(generator,
                            split,
                            includes_target=True,
                            steps=steps,
                            training=training,
                            verify=verify)

    def eval_dataset(self,
                     dataset,
                     steps: int = None,
                     training: bool = False,
                     verify: bool = False):
        raise NotImplementedError

    def eval_generator(self,
                       generator,
                       name: str,
                       includes_target: bool = False,
                       steps: int = None,
                       training: bool = False,
                       verify: bool = False):
        raise NotImplementedError

    @property
    def eval_model(self):
        if not self._eval_model:
            self._eval_model = self.init_model(training=False)
            logger.info('Created evaluation model')
            checkpoint = tf.train.latest_checkpoint(self.local_path(self.name))
            if checkpoint:
                logger.info('Found checkpoint %s', checkpoint)
                self._eval_model.load_weights(checkpoint).expect_partial()
                logger.info('Loaded weights from checkpoint')
        return self._eval_model

    @property
    def eval_dir(self):
        return self.eval_path(self.name)

    @property
    def init_dataset(self):
        raise NotImplementedError

    @property
    def init_learning_rate(self):
        raise NotImplementedError

    def init_model(self, training=True):
        raise NotImplementedError

    @property
    def init_optimizer(self):
        raise NotImplementedError

    @property
    def init_solver(self):
        raise NotImplementedError

    @property
    def init_tf_dataset(self):
        raise NotImplementedError

    @property
    def init_verifier(self):
        raise NotImplementedError

    @property
    def learning_rate(self):
        if not self._learning_rate:
            self._learning_rate = self.init_learning_rate
        return self._learning_rate

    @property
    def local_dir(self):
        return self.local_path(self.name)

    @property
    def optimizer(self):
        if not self._optimizer:
            self._optimizer = self.init_optimizer
        return self._optimizer

    def prepare_tf_dataset(self, tf_dataset):
        return tf_dataset

    def prepared_tf_dataset(self, split):
        if not split in self._prepared_tf_dataset:
            dataset = self.prepare_tf_dataset(self.tf_dataset(split))
            if self.cache_dataset:
                dataset = dataset.cache()
            if self.tf_shuffle_buffer_size:
                dataset = dataset.shuffle(self.tf_shuffle_buffer_size,
                                          reshuffle_each_iteration=False)
            dataset = dataset.batch(self.batch_size,
                                    drop_remainder=self.drop_batch_remainder)
            dataset = dataset.prefetch(2)
            self._prepared_tf_dataset[split] = dataset
        return self._prepared_tf_dataset[split]

    def run(self, stream_to_wandb: bool = False):
        train_model = self.train_model
        callbacks = self.callbacks

        if stream_to_wandb:
            wandb.init(config=self.args,
                       entity=WANDB_ENTITY,
                       name=self.name,
                       project=self.WANDB_PROJECT)
            callbacks += [
                WandbCallback(monitor=self.checkpoint_monitor, save_model=False)
            ]

        history = train_model.fit(
            self.prepared_tf_dataset('train').repeat(),
            callbacks=callbacks,
            epochs=self.initial_steps // self.val_freq +
            self.steps // self.val_freq,
            initial_epoch=self.initial_steps // self.val_freq,
            steps_per_epoch=self.val_freq,
            validation_data=self.prepared_tf_dataset('val'),
            validation_freq=1)
        history_filepath = os.path.join(self.local_path(self.name),
                                        'history.json')
        with open(history_filepath, 'w') as history_file:
            json.dump(history.history, history_file, indent=2)
            logger.info('Written training history to %s', history_filepath)
        self._eval_model = None

    def serve(self):
        raise NotImplementedError

    @property
    def str_args(self):
        result = self.args
        result['dtype_float'] = dtype_float_class_to_str[self.dtype_float]
        result['dtype_int'] = dtype_int_class_to_str[self.dtype_int]
        return result

    @property
    def temp_dir(self):
        temp_path = os.path.join(self.local_path(self.name), 'temp')
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)
        return temp_path

    def tf_dataset(self, split):
        if not self._tf_dataset:
            self._tf_dataset = self.init_tf_dataset
        return self._tf_dataset[split]

    @property
    def train_model(self):
        if not self._train_model:
            self._train_model = self.init_model(training=True)
            logger.info('Created training model')
            checkpoint = tf.train.latest_checkpoint(self.local_path(self.name))
            if checkpoint:
                logger.info('Found checkpoint %s', checkpoint)
                self._train_model.load_weights(checkpoint).expect_partial()
                logger.info('Loaded weights from checkpoint')
            self._train_model.compile(optimizer=self.optimizer)
            logger.info('Compiled training model')
        return self._train_model

    @property
    def verifier(self):
        if not self._verifier:
            self._verifier = self.init_verifier
        return self._verifier

    @classmethod
    def checkpoint_path(cls, name: str):
        return os.path.join(cls.local_path(name), 'checkpoint')

    @classmethod
    def eval_path(cls, name: str):
        return os.path.join(cls.local_path(name), 'eval')

    @classmethod
    def add_init_args(cls, parser):
        defaults = cls.get_default_args()
        if defaults['auto_version']:
            parser.add_argument('--no-auto-version',
                                action='store_false',
                                dest='auto_version')
        else:
            parser.add_argument('--auto-version', action='store_true')
        parser.add_argument('--batch-size',
                            type=int,
                            default=defaults['batch_size'])
        parser.add_argument('--no-dataset-cache',
                            action='store_false',
                            dest='cache_dataset')
        parser.add_argument('--checkpoint-monitor',
                            type=str,
                            default=defaults['checkpoint_monitor'])
        parser.add_argument('-d',
                            '--dataset',
                            dest='dataset_name',
                            default=defaults['dataset_name'])
        parser.add_argument('--no-drop-batch-remainder',
                            action='store_false',
                            dest='drop_batch_remainder')
        parser.add_argument('--dtype-float',
                            default='float32',
                            choices=dtype_int_str_to_class.keys())
        parser.add_argument('--dtype-int',
                            default='int32',
                            choices=dtype_int_str_to_class.keys())
        parser.add_argument('--steps', type=int, default=defaults['steps'])
        parser.add_argument('--initial-steps',
                            type=int,
                            default=defaults['initial_steps'])
        parser.add_argument('-n', '--name', default=defaults['name'])
        parser.add_argument('-p', '--parent', dest='parent_name', default=None)
        parser.add_argument('--no-shuffle-on-load',
                            action='store_false',
                            dest='shuffle_on_load')
        parser.add_argument('--tf-shuffle-buffer-size',
                            type=int,
                            default=defaults['tf_shuffle_buffer_size'])
        parser.add_argument('--val-freq',
                            type=int,
                            default=defaults['val_freq'])

    @classmethod
    def add_train_args(cls, parser):
        parser.add_argument('--save-to-wandb', action='store_true')
        parser.add_argument('--stream-to-wandb', action='store_true')
        parser.add_argument('-u', '--upload', action='store_true')

    @classmethod
    def add_eval_args(cls, parser):
        parser.add_argument('-n', '--name', required=True)
        parser.add_argument('-u', '--upload', action='store_true')

    @classmethod
    def cli(cls):
        parser = argparse.ArgumentParser(description='ML2 experiment')
        subparsers = parser.add_subparsers(dest='command', help='')

        train_parser = subparsers.add_parser('train', help='Training')
        cls.add_init_args(train_parser)
        cls.add_train_args(train_parser)

        eval_parser = subparsers.add_parser('eval', help='Evaluation')
        cls.add_eval_args(eval_parser)

        args = parser.parse_args()
        args_dict = vars(args)
        command = args_dict.pop('command')

        if command == 'train':
            if args_dict['parent_name']:
                parent_experiment = cls.load(args_dict['parent_name'])
                parent_args = parent_experiment.args
                parent_args.update(args_dict)
                args_dict = parent_args
            save_to_wandb = args_dict.pop('save_to_wandb')
            stream_to_wandb = args_dict.pop('stream_to_wandb')
            upload = args_dict.pop('upload')
            experiment = cls(**args_dict)
            experiment.run(stream_to_wandb=stream_to_wandb)
            experiment.save(experiment.name,
                            auto_version=False,
                            upload=upload,
                            overwrite_local=True,
                            add_to_wandb=save_to_wandb)

        elif command == 'eval':
            experiment = cls.load(args_dict['name'])
            args_dict.pop('name')
            upload = args_dict.pop('upload')
            experiment.eval(**args_dict)
            if upload:
                cls.upload(f'{experiment.name}/eval', overwrite=True)

        else:
            raise Exception("Unknown command %s", args.command)

    @classmethod
    def load_from_path(cls, path: str):
        args_filepath = os.path.join(path, 'args.json')
        if not os.path.exists(args_filepath):
            raise Exception('Could not locate arguments file %s', args_filepath)
        with open(args_filepath, 'r') as args_file:
            args = json.load(args_file)
        args['auto_version'] = False
        experiment = cls(**args)
        return experiment

    @classmethod
    def get_default_args(cls):
        default_args = {}
        for super_class in reversed(cls.mro()):
            signature = inspect.signature(super_class.__init__)
            default_args.update(
                {k: v.default for k, v in signature.parameters.items()})
        for special_arg in ['self', 'args', 'kwargs']:
            default_args.pop(special_arg)
        return default_args
