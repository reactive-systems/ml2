"""PyTorch utilities"""

import torch

str_to_pt_float_dtype = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
pt_float_dtype_to_str = {c: s for s, c in str_to_pt_float_dtype.items()}

str_to_pt_int_dtype = {"int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
pt_int_dtype_to_str = {c: s for s, c in str_to_pt_int_dtype.items()}
