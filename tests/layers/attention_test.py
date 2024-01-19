"""Tests for scaled dot-product attention and multi-head attention"""

import pytest
import tensorflow as tf

from ml2.layers.attention import MultiHeadAttention, scaled_dot_product_attention


@pytest.mark.tf
class ScaledDotProductAttentionTest(tf.test.TestCase):
    def test_single_query_no_mask(self):
        query = tf.constant([[0, 1, 0]], dtype=tf.float32)
        keys = tf.constant([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 100]], dtype=tf.float32)
        values = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
        attn, attn_weights = scaled_dot_product_attention(query, keys, values)
        expected_attn = tf.constant([[0, 1]], dtype=tf.float32)
        expected_attn_weights = tf.constant([[0, 1, 0, 0]], dtype=tf.float32)
        self.assertAllClose(attn, expected_attn)
        self.assertAllClose(attn_weights, expected_attn_weights)

    def test_multiple_queries_no_mask(self):
        queries = tf.constant([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=tf.float32)
        keys = tf.constant([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 100]], dtype=tf.float32)
        values = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
        attn, attn_weights = scaled_dot_product_attention(queries, keys, values)
        expected_attn = tf.constant([[0, 0], [0, 1], [0, 0.5]], dtype=tf.float32)
        expected_attn_weights = tf.constant(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0]], dtype=tf.float32
        )
        self.assertAllClose(attn, expected_attn)
        self.assertAllClose(attn_weights, expected_attn_weights)


@pytest.mark.tf
class MultiHeadAttentionTest(tf.test.TestCase):
    def test_shape_no_mask(self):
        multi_head_attn = MultiHeadAttention(d_embedding=512, num_heads=8)
        x = tf.random.uniform((2, 10, 512))
        attn, attn_weights = multi_head_attn(x, x, x)
        self.assertEqual(attn.shape, (2, 10, 512))
        self.assertEqual(attn_weights.shape, (2, 8, 10, 10))
