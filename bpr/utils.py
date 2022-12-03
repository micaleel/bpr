from inspect import isroutine
import logging
import os
import itertools
import random
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Optional, Tuple, Union, List, Text, Any, NamedTuple

import attr
import numpy as np
import pandas as pd
import wandb
from attr import dataclass
from sklearn.preprocessing import LabelEncoder

# types
Numeric = Union[int, float]

# constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
SPLIT_LOO = "leave_one_out"
SPLIT_STRATIFIED = "stratified"
METRIC_HR = "hit_ratio"
METRIC_PREC = "precision"


def get_rng(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    return rng


def get_rating_matrix(df, n_users, n_items, fill_value=0):
    X = np.full([n_users, n_items], fill_value).astype(np.float64)
    X[df["user_id"], df["item_id"]] = df["rating"]
    return X


def encode_labels(train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    all_df = pd.concat([train_df, test_df], axis=0)
    encoders = {
        "user_id": LabelEncoder().fit(all_df["user_id"]),
        "item_id": LabelEncoder().fit(all_df["item_id"]),
    }
    for df in [train_df, test_df]:
        for col, encoder in encoders.items():
            df[col] = encoder.transform(df[col])

    return train_df, test_df, all_df["user_id"].nunique(), all_df["item_id"].nunique()


def split_stratified(
    df, split_ratio=0.8, min_ratings=0, rng: np.random.Generator = None
):
    train, test = [], []
    rng = rng or get_rng(SEED)

    for key, X in df.groupby("user_id"):
        # discard users with less than min ratings
        if X.shape[0] < min_ratings:
            continue

        msk = rng.random(len(X)) < split_ratio
        if len(X[msk]) < 1:
            train.append(X)
            continue

        test.append(X[~msk])
        train.append(X[msk])

    return pd.DataFrame(pd.concat(train)), pd.DataFrame(pd.concat(test))


def split_loo(df, min_ratings=2, shuffle=False):
    train, test = [], []
    min_ratings = min_ratings if min_ratings else 2

    for key, X in df.groupby("user_id"):
        # discard users with less than min ratings
        if X.shape[0] < min_ratings or X.shape[0] < 2:
            continue

        if shuffle:
            X = shuffle(X)
        test.append(X[-1:])
        train.append(X[:-1])

        tr_it = set(list(X[:-1]["item_id"].unique()))
        te_it = set(list(X[-1:]["item_id"].unique()))
        try:
            assert len(X[-1:]) == 1
            assert len(tr_it.intersection(te_it)) == 0
        except:
            print(tr_it, te_it)

    return pd.DataFrame(pd.concat(train)), pd.DataFrame(pd.concat(test))


class Dataset:
    def __init__(self, train_df, test_df):
        train_df, test_df, n_users, n_items = encode_labels(train_df, test_df)
        kwds = dict(n_users=n_users, n_items=n_items)

        self.n_items = n_items
        self.n_users = n_users
        self.train_df = train_df
        self.test_df = test_df
        self.train_matrix = get_rating_matrix(self.train_df, **kwds)
        self.test_matrix = get_rating_matrix(self.test_df, **kwds)

    @property
    def train_rating_matrix(self):
        return self.train_matrix

    @property
    def test_rating_matrix(self):
        return self.test_matrix


def create_dataset(
    df,
    split_ratio=0.8,
    min_ratings=0,
    split_type="stratified",
):
    if split_type == SPLIT_STRATIFIED:
        train_df, test_df = split_stratified(
            df, split_ratio=split_ratio, min_ratings=min_ratings
        )
    elif split_type == SPLIT_LOO:
        train_df, test_df = split_loo(df, min_ratings=min_ratings)
    else:
        raise ValueError("Not implemented")

    return Dataset(train_df=train_df, test_df=test_df)


def load_dataset(name, split_ratio=0.8, min_ratings=5, split_type="stratified"):
    columns = ["user_id", "item_id", "rating"]
    columns_ml = ["user_id", "item_id", "rating", "timestamp"]
    meta = {
        "ml100k": {
            "path": os.path.join(DATA_DIR, "ml-100k", "u.data"),
            "sep": "\t",
            "columns": columns_ml,
        },
        "ml1m": {
            "path": os.path.join(DATA_DIR, "ml-1m", "ratings.dat"),
            "sep": ",",
            "columns": columns_ml,
        },
        "tripadvisor": {
            "path": os.path.join(DATA_DIR, "tripadvisor", "tripadvisor.csv"),
            "sep": ",",
            "columns": columns,
        },
        "yelp": {
            "path": os.path.join(DATA_DIR, "yelp", "yelp.csv"),
            "sep": ",",
            "columns": columns,
        },
    }
    params = meta[name]
    df = pd.read_csv(params["path"], names=params["columns"], sep=params["sep"])
    return create_dataset(
        df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type
    )


def load_ml100k(path=None, split_ratio=0.8, min_ratings=0, split_type="stratified"):
    if not path:
        path = os.path.join(DATA_DIR, "ml-100k", "u.data")

    df = pd.read_csv(
        path, names=["user_id", "item_id", "rating", "timestamp"], sep="\t"
    )
    df.drop("timestamp", 1)

    return create_dataset(
        df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type
    )


def load_ml1m(path=None, split_ratio=0.8, min_ratings=0, split_type="stratified"):
    if not path:
        path = os.path.join(DATA_DIR, "ml-1m", "ratings.dat")

    df = pd.read_csv(
        path, names=["user_id", "item_id", "rating", "timestamp"], sep=",", skiprows=1
    )
    df.drop("timestamp", 1)

    return create_dataset(
        df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type
    )


def calculate_metrics(
    recommendations_list, actual_rating_matrix, get_average=False, ndcg=False
):
    # TODO Cythonize
    max_hits = np.sum(actual_rating_matrix > 0)
    rateable = np.count_nonzero(np.sum(actual_rating_matrix, axis=1))
    top_n = recommendations_list.shape[1]
    tpfn = rateable * top_n

    n_tp = 0
    __ux = []
    for user in range(recommendations_list.shape[0]):
        _ux = 0
        for item in recommendations_list[user]:
            if actual_rating_matrix[user, item] > 0:
                n_tp += 1
                _ux += 1
        if get_average:
            __ux.append(_ux / recommendations_list.shape[1])
        # if ndcg:
        # assuming recommendations_list is sorted

    precision = n_tp / tpfn
    recall = n_tp / max_hits

    result = {
        "PRECISION@{}".format(top_n): precision,
        "RECALL@{}".format(top_n): recall,
    }
    if not get_average:
        return result
    return result, __ux


def load_ml100k_new():
    data = pd.read_csv(
        os.path.join(DATA_DIR, "ml-100k", "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    data["rating"] = 1
    data["user_id"] = LabelEncoder().fit_transform(data["user_id"])
    data["item_id"] = LabelEncoder().fit_transform(data["item_id"])
    return data


def split(
    data: pd.DataFrame,
    chrono: bool = True,
    loo: bool = True,
    min_ratings: int = 0,
    shuffle: bool = False,
    train_size=0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = [], []  # List[np.ndarray]

    for user_id, df in data.groupby("user_id", sort=False):
        if chrono:
            if "timestamp" in df.columns:
                X = df.sort_values("timestamp").values
            else:
                raise ValueError("timestamp column missing in DataFrame")
        else:
            X = df.values

        if X.shape[0] < min_ratings:
            continue

        if shuffle:
            np.random.shuffle(X)  # TODO Use default_rng()

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        mask = np.full(n_samples, fill_value=True, dtype=np.bool)

        if loo:
            mask[-1] = False
        else:
            train_idxs = np.random.choice(
                a=indices, size=int(n_samples * train_size), replace=False
            )
            mask[train_idxs] = False
        train.append(X[mask])
        test.append(X[np.logical_not(mask)])
    return np.concatenate(train), np.concatenate(test)


def partial_ranking_mask(
    train_matrix,
    test_matrix,
    n_negatives=50,
    rng: np.random._generator.Generator = None,
):
    """Returns bool test_matrix with negative samples"""
    rng = rng or get_rng(SEED)
    mask = test_matrix > 0
    not_rated = (train_matrix + test_matrix) == 0
    items = np.arange(train_matrix.shape[1])
    n_test_rated = np.sum((test_matrix > 0).astype(np.int32), axis=1)
    n_not_rated = np.sum((not_rated > 0).astype(np.int32), axis=1)  #
    # unrated items have equal probabilities of being selected.
    pick_probs = (
        not_rated.astype(np.int32) / np.sum(not_rated.astype(np.int32), axis=1)[:, None]
    )
    for user in range(train_matrix.shape[0]):
        negatives = rng.choice(
            items,
            size=min(n_not_rated[user], n_test_rated[user] * n_negatives),
            replace=False,
            p=pick_probs[user],
        )
        mask[user, negatives] = 1
    return mask


class EpochPrinter:
    def __init__(self, epochs, prefix="Epoch"):
        self.epochs = epochs
        self.prefix = prefix
        if epochs <= 10:
            self.checkpoints = list(range(epochs))
        else:
            self.checkpoints = [
                int(z) for z in np.linspace(start=0, stop=epochs, num=11, endpoint=True)
            ]
        self.checkpoints.insert(0, 1)
        self.checkpoints.append(epochs)

    def __call__(self, epoch, log=None):
        # print(epoch)
        if epoch + 1 in self.checkpoints:
            message = "{} {:>3}".format(self.prefix, epoch + 1)
            if log is not None:
                log.info(message)
            else:
                print(message)


@dataclass
class ExperimentConfig:
    """
    Configuration options for experiments
    """

    biased: bool = True
    blend: float = 0
    diff_limit: int = 0
    epoch_decayed: bool = False
    epochs: int = 1000
    item_probs_uniform: bool = False
    se_0: Numeric = 0
    se_z: Optional[Numeric] = None
    wand_project: str = "201204"
    warm_up: Optional[int] = None
    warm_up_biased: Optional[bool] = False
    window_size: int = 0

    @property
    def name(self) -> str:
        if not self.biased:
            return f"e{self.epochs}-baseline"
        else:
            output = []
            output.append(f"b{self.blend}")
            output.append(f"dl{self.diff_limit}")
            output.append(f"ed{str(self.epoch_decayed)[0]}")
            output.append(f"ipu{str(self.item_probs_uniform)[0]}")
            output.append(f"se0{self.se_0:g}".replace("+", ""))
            if self.se_z:
                output.append(f"sez{self.se_z:g}".replace("+", ""))
            output.append(f"wb{str(self.warm_up_biased)[0]}")
            output.append(f"ws{self.window_size}")
            output.append(f"wu{self.warm_up}")
            return "-".join(output)

    @property
    def is_valid(self):
        valid = True
        is_baseline = not self.biased

        if is_baseline:
            # all parameters for controlling bias should be 0
            valid = all(
                p == 0
                for p in (self.window_size, self.warm_up, self.blend, self.diff_limit)
            )
            valid = valid and self.se_0 == 1
            valid = valid and self.se_z is None

            valid = valid and all(
                p is False for p in (self.warm_up_biased, self.epoch_decayed)
            )
            valid = valid and self.item_probs_uniform

        if self.biased:
            valid = all(p > 0 for p in (self.se_0, self.window_size, self.warm_up))

            if self.se_0 is None and self.se_z is not None:
                valid = False

        return valid

    def baseline_conf(self, epochs: Optional[int] = None) -> "ExperimentConfig":
        return ExperimentConfig(
            biased=False,
            blend=0,
            epoch_decayed=False,
            epochs=epochs or self.epochs,
            item_probs_uniform=True,
            se_0=1,
            se_z=None,
            wand_project=self.wand_project,
            warm_up=0,
            warm_up_biased=False,
            window_size=0,
        )

    def asdict(self) -> dict:
        result = attr.asdict(self)
        return result

    @classmethod
    def generate(
        cls,
        biased: List[bool],
        blend: List[float],
        diff_limit: List[float],
        epochs_decayed: List[bool],
        item_probs_uniform: List[bool],
        se_0: List[int],
        se_z: List[int],
        warm_up: List[int],
        warm_up_biased: List[bool],
        window_size: List[int],
    ):
        def _list(x: Any):
            try:
                _ = iter(x)
                return list(x)
            except TypeError:
                return [x]

        # Generate parameter space
        parameters = [
            ExperimentConfig(
                biased=_biased,
                blend=_blend,
                diff_limit=_diff_limit,
                epoch_decayed=_epochs_decayed,
                item_probs_uniform=_item_probs_uniform,
                se_0=_se_0,
                se_z=_se_z,
                warm_up=_warm_up,
                warm_up_biased=_warm_up_biased,
                window_size=_window_size,
            )
            for (
                _biased,
                _blend,
                _diff_limit,
                _epochs_decayed,
                _item_probs_uniform,
                _se_0,
                _se_z,
                _warm_up,
                _warm_up_biased,
                _window_size,
            ) in itertools.product(
                _list(biased),
                _list(blend),
                _list(diff_limit),
                _list(epochs_decayed),
                _list(item_probs_uniform),
                _list(se_0),
                _list(se_z),
                _list(warm_up),
                _list(warm_up_biased),
                _list(window_size),
            )
        ]  # type: List[ExperimentConfig]
        parameters = [p for p in parameters if p.is_valid]
        return parameters


@contextmanager
def timed_op(msg: str, logger=None, level: Union[int, Text] = None):
    try:
        t0 = time.time()
        yield
    finally:
        if logger:
            duration = timedelta(time.time() - t0)
            msg = f"{msg} ({str(duration)})"
            if level == logging.DEBUG:
                logger.debug(msg)
            elif level == logging.INFO:
                logger.info(msg)
            else:
                raise NotImplementedError


@contextmanager
def wandb_logger(
    name: str, wandb_project: str, wandb_config: dict, reinit: bool = True
):
    run = wandb.init(config=wandb_config, project=wandb_project, reinit=reinit)
    run.name = name

    try:
        yield run
    finally:
        run.finish()


def asdict(instance):
    import inspect

    attributes = inspect.getmembers(instance, lambda a: not (inspect.isroutine(a)))
    valid = {}
    for a in attributes:
        if not (a[0].startswith("__") and a[0].endswith("__")):
            if isinstance(a[1], (int, float, str)):
                valid[a[0]] = a[1]
    return dict(valid)
