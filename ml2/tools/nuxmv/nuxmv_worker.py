"""nuXmv worker"""

import ray

from .nuxmv_wrapper import nuxmv_wrapper_dict


@ray.remote
def nuxmv_worker(spec: dict, circuit: str, realizable: bool, strix_path, temp_dir, timeout):
    return nuxmv_wrapper_dict(spec, circuit, realizable, strix_path, temp_dir, timeout)
