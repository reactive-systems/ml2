"""Statistics utilities"""

import json
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def stats_from_counts(counts: dict):
    stats_dict = {}
    for key in counts:
        data = counts[key]
        stats_dict[key] = {}
        stats_dict[key]["max"] = max(data)
        stats_dict[key]["min"] = min(data)
        stats_dict[key]["avg"] = np.average(data).item()
        stats_dict[key]["median"] = np.median(data).item()
        stats_dict[key]["bins"] = np.bincount(data).tolist()
    return stats_dict


def write_stats(stats, file):
    with open(file, "w") as stats_file:
        json.dump(stats, stats_file, indent=2)
        logging.info("Statistics written to %s", file)


def plot_stats(stats, file):
    _, axes = plt.subplots(1, 5, figsize=(15, 3), dpi=100)
    for axis, key in zip(axes.flatten(), stats):
        max_value = stats[key]["max"]
        bins = stats[key]["bins"]
        axis.bar(range(max_value + 1), bins, width=0.7, align="center")
        axis.set_title(key.lower())
        axis.set_ylim(0, max(bins) + 1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(file)
    logging.info("Statistics plotted to %s", file)
