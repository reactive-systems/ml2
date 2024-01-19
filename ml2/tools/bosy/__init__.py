from ...utils import is_ray_available
from .bosy import BoSy

if is_ray_available():
    from .bosy_worker import add_bosy_args, bosy_worker_fn, bosy_worker_fn_dict
