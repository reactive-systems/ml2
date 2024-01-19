"""NumPy utilities"""

import numpy as np

str_to_np_float_dtype = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
np_float_dtype_to_str = {c: s for s, c in str_to_np_float_dtype.items()}

str_to_np_int_dtype = {"int16": np.int16, "int32": np.int32, "int64": np.int64}
np_int_dtype_to_str = {c: s for s, c in str_to_np_int_dtype.items()}
