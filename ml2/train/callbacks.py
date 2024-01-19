"""Callbacks"""

from ray import tune
import tensorflow as tf


class TuneReporterCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.iteration = 0
        super().__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(
            keras_info=logs,
            mean_accuracy=logs.get("accuracy_per_sequence"),
            mean_loss=logs.get("loss"),
        )
