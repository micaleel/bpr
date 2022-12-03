from datetime import datetime
from typing import Dict, List, Optional, Union

import os
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from contextlib import contextmanager
from typing import Optional, Tuple, Union, List, Text, Any, NamedTuple
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# types
CSRMatrix = scipy.sparse.csr_matrix
RandomGenerator = Union[np.random.Generator, np.random.RandomState]


def trim(df, min_size=10, copy=True):
    users = df["user_id"].value_counts()
    s = users[users >= min_size].index.tolist()
    subset = df[df["user_id"].isin(s)]
    return subset.copy() if copy else subset


def get_random_state(state: Any) -> RandomGenerator:
    state = state or 42
    if isinstance(state, (np.random.RandomState, np.random.Generator)):
        return state
    return np.random.RandomState(state)


def _load_edoardo(path):
    if not os.path.exists(path):
        raise ValueError("path does not exist")

    if not os.path.isdir(path):
        raise ValueError("path should be a directory")

    if path.endswith("/"):
        path = path[:-1]

    def _load_split(fpath):
        return (
            pd.read_csv(fpath)
            .drop("Unnamed: 0", axis="columns")
            .rename(columns={"userID": "user_id", "itemID": "item_id"})
            .assign(rating=lambda _: 1)
        )

    df_train = _load_split(path + "/train.csv")
    df_validation = _load_split(path + "/val.csv")
    df_test = _load_split(path + "/test.csv")
    return pd.concat([df_train, df_validation, df_test])


load_amazon = _load_edoardo
load_gowalla = _load_edoardo
load_lastfm = _load_edoardo


def load_amazon_electronics(path):
    data = pd.read_csv(path)
    return data


def load_ml1m(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df


def load_ml100k(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df


def load_tripadvisor(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=1,
        names=["user_id", "item_id", "rating"],
    )
    return df


def load_yelp(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=1,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df


def load_citeulike(path: str) -> pd.DataFrame:
    file_lines = []
    with open(path) as file:
        file_lines = file.readlines()

    item_ids_str = set()
    for line in file_lines:
        item_ids_str.update(line.split())

    encoder = LabelEncoder().fit(list(item_ids_str))

    users, items = [], []
    for user, line in enumerate(file_lines):
        item_ids_int = encoder.transform(line.split())
        users.extend([user] * len(item_ids_int))
        items.extend(item_ids_int)

    df = pd.DataFrame({"user_id": users, "item_id": items})
    df["user_id_item_id"] = (
        df["user_id"].astype(str).str.cat(df["item_id"].astype(str), sep="#")
    )

    tmp_df = (
        df["user_id_item_id"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "user_id_item_id", "user_id_item_id": "rating"})
    )

    return (
        pd.merge(tmp_df, df, on="user_id_item_id")
        .drop_duplicates()
        .drop(columns=["user_id_item_id"])
    )


def load_mind(path):
    df = pd.read_csv(
        path,
        sep="\t",
        names=["impression_id", "user_id", "time", "history", "impressions"],
    ).drop(columns=["time", "impression_id"])

    df.dropna(inplace=True)

    user_ids, item_ids = [], []
    for row in df.itertuples():
        user_id = getattr(row, "user_id")
        clicked = set(getattr(row, "history").split())
        for impression in getattr(row, "impressions"):
            if impression.endswith("-1"):
                clicked.add(impression[:-2])
        user_ids.extend([user_id] * len(clicked))
        item_ids.extend(list(clicked))

    df_a = pd.DataFrame({"user_id": user_ids, "item_id": item_ids})
    df_a["user_id_item_id"] = (
        df_a["user_id"].astype(str).str.cat(df_a["item_id"].astype(str), sep="#")
    )

    df_b = (
        df_a["user_id_item_id"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "user_id_item_id", "user_id_item_id": "rating"})
    )

    final_df = (
        pd.merge(df_b, df_a, on="user_id_item_id")
        .drop_duplicates()
        .drop(columns=["user_id_item_id"])
    )
    final_df["user_id"] = LabelEncoder().fit_transform(final_df["user_id"])
    final_df["item_id"] = LabelEncoder().fit_transform(final_df["item_id"])
    return final_df[["user_id", "item_id", "rating"]]


def sample(df, size, priority="random", min_size=10) -> pd.DataFrame:
    assert isinstance(size, (int, float))
    assert priority in (
        "random",
        "high",
        "low",
    ), "priority must be one of 'random', 'high', or 'low'"
    assert "user_id" in df.columns, "Failed to find 'user_id' column"

    user_ids = df["user_id"].value_counts()
    user_ids = user_ids[user_ids > min_size].sort_values()

    if isinstance(size, float):
        assert 0 < size < 1, "size must be in range (0, 1)"
        size = int(len(user_ids) * size)

    chosen = None
    if size > len(user_ids):
        chosen = user_ids.index
    elif priority == "random":
        chosen = user_ids.sample(size).index
    elif priority == "low":
        chosen = user_ids.tail(size).index
    elif priority == "high":
        chosen == user_ids.head(size).index
    else:
        raise ValueError("Unknown priority: %s" % priority)

    return df[df["user_id"].isin(chosen)]


class Dataset:
    def __init__(
        self,
        df: pd.DataFrame,
        train_size: float = 0.8,
        loo: bool = False,
        chrono: bool = False,
        min_ratings=5,
    ) -> None:
        self.train_size = train_size
        self.loo = loo
        self.chrono = chrono
        self.min_ratings = min_ratings

        self.n_users = df["user_id"].nunique()
        self.n_items = df["item_id"].nunique()

        self.df = df.copy()
        self.df["user_id"] = LabelEncoder().fit_transform(df["user_id"])
        self.df["item_id"] = LabelEncoder().fit_transform(df["item_id"])

        self.train_matrix, self.test_matrix = self.split(
            df=self.df,
            train_size=train_size,
            loo=loo,
            chrono=chrono,
            min_ratings=min_ratings,
        )
        self.__negatives = None  # type: Optional[np.ndarray]
        self._item_simlarities = None  # type: Optional[np.ndarray]

        # create full user-item matrix
        ratings = np.zeros(shape=(self.n_users, self.n_items))
        ratings[self.df["user_id"], self.df["item_id"]] = 1
        self.ratings_ui = csr_matrix(ratings)

        del ratings

    @property
    def item_similarities(self) -> np.ndarray:
        """Returns a pairwise matrix of item-item similarities"""
        if self._item_simlarities is None:
            self._item_simlarities = cosine_similarity(
                self.train_matrix.T, dense_output=False
            )
        return self._item_simlarities

    @property
    def negatives(self):
        if self.__negatives is None:
            self.__negatives = csr_matrix(
                1 - (self.train_matrix.A + self.test_matrix.A)
            )
        return self.__negatives

    @property
    def unrated(self):
        return self.negatives

    def __repr__(self):
        return (
            f"Dataset("
            f"chrono={self.chrono}, "
            f"loo={self.loo}, "
            f"min_ratings={self.min_ratings}, "
            f"n_items={self.n_items:,}, "
            f"n_ratings={len(self.df):,}, "
            f"n_users={self.n_users:,}, "
            f"train_size={self.train_size:.1f})"
        )

    def split(
        self,
        df: pd.DataFrame,
        train_size: float,
        loo: bool = False,
        chrono: bool = True,
        min_ratings=5,
        rs: Optional[RandomGenerator] = None,
    ):
        rs = get_random_state(rs)
        use_chrono = chrono and "timestamp" in df.columns
        train_matrix = np.zeros(shape=(self.n_users, self.n_items))
        test_matrix = np.zeros(shape=(self.n_users, self.n_items))

        for user_id, u_df in df.groupby("user_id", sort=False):
            if min_ratings is not None and len(df) < min_ratings:
                continue

            if use_chrono:
                item_ids = u_df.sort_values("timestamp")["item_id"].values
            else:
                item_ids = u_df["item_id"].values
                rs.shuffle(item_ids)

            if loo:
                item_ids_train, item_ids_test = item_ids[:-1], item_ids[-1]
            else:
                train_idx = int(item_ids.shape[0] * train_size)
                item_ids_train = item_ids[:train_idx]
                item_ids_test = item_ids[train_idx:]

            train_matrix[user_id, item_ids_train] = 1
            test_matrix[user_id, item_ids_test] = 1

        self.__negatives = csr_matrix(1 - (train_matrix + test_matrix))

        return csr_matrix(train_matrix), csr_matrix(test_matrix)


def evaluate(recommendations: np.ndarray, test_matrix: scipy.sparse.csr.csr_matrix):
    test_matrix_csr = test_matrix
    n_interactions = test_matrix_csr.nnz
    n_users = test_matrix.shape[0]
    n_recommended_items = n_users * recommendations.shape[1]
    n_hits = 0
    for user in range(recommendations.shape[0]):
        tp = set(recommendations[user]).intersection(test_matrix_csr[user].indices)
        n_hits += len(tp)
    precision = n_hits / n_recommended_items
    recall = n_hits / n_interactions
    top_n = recommendations.shape[1]
    result = {
        "PRECISION@{}".format(top_n): precision,
        "RECALL@{}".format(top_n): recall,
    }
    return result


def plot_metrics(
    losses: Dict[str, List[int]],
    precisions: Dict[str, List[int]],
    title: str = None,
    fname: str = None,
    lw=4,
):
    assert isinstance(losses, dict)
    assert isinstance(precisions, dict)
    if not title:
        now = datetime.now()
        title = now.strftime("%H:%M:%S")

    aliases = {"RND": "Default", "SIM": "Similarity"}
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    for label, losses_ in losses.items():
        ax0.plot(losses_, label=aliases.get(label, label), lw=lw)
    ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Loss")
    ax0.set_ylim(0, 1)
    ax0.grid(True, color="gainsboro", ls="dotted", lw=0.5)
    for label, precisions_ in precisions.items():
        ax1.plot(precisions_, label=label, lw=lw)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Precision@10")
    ax1.grid(True, color="gainsboro", ls="dotted", lw=0.5)
    ax1.set_ylim(0.15, 0.4)
    plt.suptitle(title, fontsize="xx-small")
    ax1.legend(ncol=2)
    sns.despine()
    fig.tight_layout()
    fname = fname or "myfig.png"
    fig.savefig(fname, dpi=400)


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
    name: str,
    wandb_project: str,
    wandb_config: dict,
    reinit: bool = True,
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
