import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re

import sys

sys.path.extend(["../bpr", "../"])

from ray import tune
from ray.tune.schedulers import ASHAScheduler

search_space = {
    #     "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    #     "reg": tune.uniform(0.01, 0.9)
    "n_factors": tune.grid_search(list(range(10, 50)))
}


def train_func(config):
    import sys

    sys.path.append("../bpr")
    import os

    print("\n\n", os.path.abspath(os.curdir), "\n\n")
    from train_simple import train_bpr, RandomSampler
    from functools import partial
    from utils import load_ml100k

    data = load_ml100k("/home/khalil/projects/bpr/data/ml-100k/u.data")
    n_negatives = 10
    sampler = RandomSampler(data, n_negatives * len(data.train_df))

    train_bpr(data=data, sampler=sampler, n_factors=config["n_factors"])


if __name__ == "__main__":
    analysis = tune.run(train_func, config=search_space)
