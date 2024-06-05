"""TensorFlow Transformer metrics"""

import tensorflow as tf


class TransformerAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name="acc", dtype_float: tf.DType = tf.float32, pad_id: int = 0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dtype_float = dtype_float
        self.pad_id = pad_id
        self.acc_mean = tf.keras.metrics.Mean("acc")
        self.acc_per_seq_mean = tf.keras.metrics.Mean("acc_per_seq")

    def update_state(self, y_true, y_pred, sample_weight=None):
        weights = tf.cast(tf.not_equal(y_true, self.pad_id), self.dtype_float)
        outputs = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        y_true = tf.cast(y_true, tf.int32)

        # accuracy
        correct_predictions = tf.cast(tf.equal(outputs, y_true), self.dtype_float)
        self.acc_mean.update_state(correct_predictions, weights)

        # accuracy per sequence
        incorrect_predictions = tf.cast(tf.not_equal(outputs, y_true), self.dtype_float) * weights
        correct_sequences = 1.0 - tf.minimum(1.0, tf.reduce_sum(incorrect_predictions, axis=-1))
        self.acc_per_seq_mean.update_state(correct_sequences, tf.constant(1.0))

    def reset_state(self):
        self.acc_mean.reset_state()
        self.acc_per_seq_mean.reset_state()

    def result(self):
        return {
            "acc": self.acc_mean.result(),
            "acc_per_seq": self.acc_per_seq_mean.result(),
        }
