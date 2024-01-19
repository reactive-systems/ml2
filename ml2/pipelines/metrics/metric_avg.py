"""Metric average"""

import json
import numpy as np
import os
from typing import Any, Dict

from .metric import Metric


class MetricAvg:
    def __init__(self, name: str = "metric-avg") -> None:
        self.name = name
        self._metrics = []

    def add_metric(self, metric: Metric) -> Any:
        self._metrics.append(metric)

    def compute_dict(self) -> Dict[str, Any]:
        result = {}
        computations = [m.compute_dict() for m in self._metrics]
        if len(computations) == 0:
            return result
        for key in computations[0]:
            result[key] = {}
            values = [c[key] for c in computations]
            result[key]["mean"] = float(np.mean(values))
            result[key]["std"] = float(np.std(values))
            result[key]["max"] = float(np.amax(values))
            result[key]["min"] = float(np.amin(values))
        return result

    def reset(self) -> None:
        self._metrics = []

    def save_to_path(self, path: str) -> None:
        filepath = os.path.join(path, self.name + ".json")
        with open(filepath, "w") as metric_file:
            json.dump(self.compute_dict(), metric_file, indent=2)
