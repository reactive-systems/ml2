"""Statistics utilities"""

import json
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


class HasX:
    def __init__(self, x: int) -> None:
        self.x = x


def f() -> HasX:
    o = 3
    o.x = 5
    return o


def stats_from_counts(counts: dict, default: int = 0):
    stats_dict = {}
    for key in counts:
        data = counts[key]
        stats_dict[key] = {}
        stats_dict[key]["bins"] = np.bincount(data).tolist()
    return {k: {**v, **stats_dict[k]} for k, v in stats(items=counts, default=default).items()}


def stats(items: dict, default: int = 0):
    stats_dict = {}
    for key in items:
        data = items[key]
        stats_dict[key] = {}
        stats_dict[key]["max"] = max(data) if data else default
        stats_dict[key]["min"] = min(data) if data else default
        stats_dict[key]["avg"] = np.average(data).item() if data else default
        stats_dict[key]["median"] = np.median(data).item() if data else default
    return stats_dict


def write_stats(stats, file):
    directory = os.path.split(file)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file, "w") as stats_file:
        json.dump(stats, stats_file, indent=2)
        logging.info("Statistics written to %s", file)


def plot_stats(stats, file):
    directory = os.path.split(file)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    _, axes = plt.subplots(
        math.ceil(len(stats) / 3), 3, figsize=(15, 3.3 * math.ceil(len(stats) / 3)), dpi=100
    )
    for axis, key in zip(axes.flatten(), stats):
        bins = stats[key]["bins"]
        axis.bar(range(len(bins)), bins, width=0.7, align="center")
        axis.set_title(key.lower())
        axis.set_ylim(0, max(bins) + 1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(file)
    logging.info("Statistics plotted to %s", file)
