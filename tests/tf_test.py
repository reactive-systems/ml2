"""TensorFlow tests"""

import pytest
import tensorflow as tf


# in some cases we observed numerical issues where for example the result of a dense layer depends on one or two vectors being passed
@pytest.mark.tf
class TFNumericIssueTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.layer = tf.keras.layers.Dense(16)

    def test_numeric_issue(self):
        v1 = tf.random.normal([16])
        v2 = tf.random.normal([16])
        i1 = tf.stack([v1])
        i2 = tf.stack([v1, v2])
        o1 = self.layer(i1)
        o2 = self.layer(i2)
        self.assertAllEqual(o1[0], o2[0])
