"""TensorFlow utilities"""

import tensorflow as tf

str_to_tf_float_dtype = {"float16": tf.float16, "float32": tf.float32, "float64": tf.float64}
tf_float_dtype_to_str = {c: s for s, c in str_to_tf_float_dtype.items()}

str_to_tf_int_dtype = {"int16": tf.int16, "int32": tf.int32, "int64": tf.int64}
tf_int_dtype_to_str = {c: s for s, c in str_to_tf_int_dtype.items()}
