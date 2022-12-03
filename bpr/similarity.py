import faulthandler
import itertools
import math
import multiprocessing as mp
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import groupby
from pprint import pprint
from re import I
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    NamedTuple,
    Optional,
    Text,
    Tuple,
    Union,
)

faulthandler.enable()
import numpy as np
import pandas as pd
import scipy
from numpy.core.fromnumeric import shape, sort
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import wandb

from kutils import *

try:
    from ._bpr import bpr_update, negative_probs_for_similar_items, popularity_to_probs
    from .kutils import Dataset, evaluate, plot_metrics
except ImportError:
    from _bpr import bpr_update, negative_probs_for_similar_items, popularity_to_probs
    from kutils import Dataset, evaluate, plot_metrics


np.random.seed(42)

# Constants
EMPTY_CLUSTER_CELL = -1
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.01
DEFAULT_N_CLUSTERS = 5
DEFAULT_N_FACTORS = 10
DEFAULT_N_NEGATIVES = 10
DEFAULT_RANDOM_STATE = get_random_state(42)

SAMPLER_DEFAULT = "default"
SAMPLER_DIVUSER = "divu"
SAMPLER_SIMITEM = "simi"
SAMPLER_NAMES = [
    SAMPLER_DEFAULT,
    SAMPLER_DIVUSER,
    SAMPLER_SIMITEM,
]

WANDB_PROJECT = "{today}-{dataset}".format(
    today=datetime.now().strftime("%Y%m%d")[2:], dataset="ml-100k"
)
WANDB_PROJECT = WANDB_PROJECT.replace("ml-100k", "tripadvisor")


TripletCreator = Union[
    Callable[[Dataset, int, int, Dict[int, List[int]]], np.ndarray],
    Callable[[Dataset, int, int], np.ndarray],
]

NUMPY_INT_DTYPE = np.int32
NUMPY_FLOAT_DTYPE = np.float64
NUM_THREADS = os.cpu_count()

DECAY_EXPONENTIAL = "ex"
DECAY_LINEAR = "ln"
DECAY_LOGARITHMIC = "lg"
DECAY_NONE = "no"
DECAY_QUADRATIC = "qd"
DECAY_MODES = [
    DECAY_EXPONENTIAL,
    DECAY_LINEAR,
    DECAY_LOGARITHMIC,
    DECAY_NONE,
    DECAY_QUADRATIC,
]

# types
WandBLogger = wandb.sdk.wandb_run.Run
RandomGenerator = Union[np.random.Generator, np.random.RandomState]


class UserClusters:
    def __init__(self, matrix, lookup, sizes=None):
        """Constructor

        Args:
            matrix (np.ndarray): matrix of clusters, rows contain similar users
            lookup (np.ndarray): cluster labels indexed by user IDs
            sizes (np.ndarray, optional)
        """
        self.matrix = matrix
        self.lookup = lookup
        self.cluster_sizes = sizes

    def compute_samples_per_cluster(self, n_samples: int):
        """Gets the fraction of samples contributed by each cluster

        Returns:
            np.ndarray
        """
        n_users = np.sum(self.cluster_sizes)
        return n_samples * (self.cluster_sizes / n_users)

    @staticmethod
    def cluster_users(
        train_matrix,
        n_clusters,
        random_state=None,
        fill=True,
    ):
        """Group users into clusters based on their interaction profiles

        Args:
            train_matrix (CSRMatrix): trainset, rows are users and columns are items
            n_clusters (int): number of clusters
            random_state (RandomGenerator, optional): random generator
            fill (bool, optional): whether to pad clusters with random users.
                Shorter clusters are padded with `EMPTY_CLUSTER_CELL`. Defaults to True

        Returns:
            UserCluster
        """
        random_state = get_random_state(random_state)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            train_matrix
        )

        uniques, counts = np.unique(kmeans.labels_, return_counts=True)
        cluster_sizes = dict(zip(uniques, counts))
        max_cluster_size = max(cluster_sizes.values())

        matrix = np.full(
            shape=[len(cluster_sizes), max_cluster_size],
            fill_value=EMPTY_CLUSTER_CELL,
            dtype=np.int32,
        )

        # Create matrix where users in similar clusters share the same row
        for cluster, size in cluster_sizes.items():
            users = np.argwhere(kmeans.labels_ == cluster).reshape(-1)
            pad_size = max_cluster_size - size
            if pad_size != 0:
                users = np.hstack([users, np.full(pad_size, EMPTY_CLUSTER_CELL)])
            matrix[cluster] = users

        if fill:
            for column in range(max_cluster_size - 1, 0, -1):
                # Find empty values in current column
                empty_args = np.argwhere(matrix[:, column] == EMPTY_CLUSTER_CELL)[:, 0]

                # Get user IDs from column preceding the current one
                nonempty_vals = np.where(matrix[:, column - 1] != EMPTY_CLUSTER_CELL)[0]

                # Fill empty values with random samples from previous column
                candidates = np.random.choice(
                    size=empty_args.shape[0], a=nonempty_vals, replace=True
                )
                matrix[empty_args, column] = candidates
        cluster_sizes_arr = np.zeros(len(cluster_sizes))
        cluster_sizes_arr[list(cluster_sizes.keys())] = list(cluster_sizes.values())
        return UserClusters(
            matrix=matrix, lookup=kmeans.labels_, sizes=cluster_sizes_arr
        )


class UserSampler:
    def __init__(
        self,
        cluster_sizes,
        clusters,
        pool_size=45,
        weights=None,
    ):
        """Constructor

        Args:
            cluster_sizes (np.ndarray): ordered list of cluster sizes
            clusters (np.ndarray): user clusters, one row per cluster.
                Since clusters are of different lengths, shorter rows
                are padded with `EMPTY_CLUSTER_CELL`
            pool_size (int):
            weights (np.ndarray, optional): array of user profile sizes
                indexed by user IDs, e.g. weights[0] gets the number of
                interactions for user 0.
        """
        self.cluster_sizes = cluster_sizes.astype(np.int32)
        self.clusters = clusters  # type: np.ndarray
        self.pool_size = pool_size

        if weights is None:
            # Make users have equal weights
            user_ids = clusters[clusters != EMPTY_CLUSTER_CELL]
            self.weights = np.full(len(user_ids), 1)
        else:
            if isinstance(weights, list):
                self.weights = np.asarray(weights, dtype=np.int32)
            elif isinstance(weights, np.ndarray):
                self.weights = weights.astype(np.int32)
            else:
                raise ValueError("weights must be an integer NumPy array")

        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        self.cluster_ids = np.arange(self.n_clusters)

        # Array of column indexes/pointers each tracking the next user
        # to be sampled from a cluster.
        self.cluster_ptrs = np.zeros(self.n_clusters, dtype=np.int32)
        self.pool = None
        self.pool_ptr = 0

    def _update_pool(self):
        """
        Choose and cache users across clusters in a round-robin manner
        """
        users = []
        loops = math.ceil(self.pool_size / self.n_clusters)
        for i in range(loops):
            users.append(self.clusters[self.cluster_ids, self.cluster_ptrs])
            self.cluster_ptrs = (self.cluster_ptrs + 1) % self.cluster_sizes
        self.pool = np.concatenate(users)
        remainder = len(self.pool) - self.pool_size

        # Adjust cluster pointers if we sample more users than required
        for i in range(remainder):
            self.cluster_ptrs[i] = (self.cluster_ptrs[i] - 1) % self.cluster_sizes[i]

        self.pool = self.pool[: self.pool_size]

    def draw_samples(self, n_samples_expected=914):
        """Sample users that would provide `n_samples_expected` triplets

        Triplets are produced by pairing every item _liked_ by a user
        with a number of items unseen by the user.

        Args:
            n_samples_expected (int): number of samples to be contributed by users

        Returns:
            List[int]: a sequence of user IDs
                The order of the user IDs is important as it dictactes the
                content of training mini-batches
        """
        users, n_samples = [], 0

        if self.pool is None:
            self._update_pool()

        while n_samples < n_samples_expected:
            n_remaining = n_samples_expected - n_samples

            # Compute pool size based on weights
            items_in_pool = self.pool[self.pool_ptr :]
            weights_cumsum = self.weights[items_in_pool].cumsum()
            n_samples_in_pool = weights_cumsum[-1]

            if n_samples_in_pool - n_remaining > 0:
                idx = np.searchsorted(weights_cumsum, n_remaining)
                if idx == 0:
                    # Samples from the first user in the pool together with the
                    # samples collected so far will exceed `n_samples_expected`.
                    # Yet we take the first user in the pool but trim the excess
                    idx = 1

                users_pooled = self.pool[self.pool_ptr : self.pool_ptr + idx]
                self.pool_ptr += len(users_pooled)
            else:
                # Take everything in the pool & adjust the pool pointer to flag
                # the need to refill the pool.
                users_pooled = self.pool[self.pool_ptr :]
                self.pool_ptr = len(self.pool)

            users.extend(users_pooled)

            # Refill the pool, if necessary
            if self.pool_ptr >= len(self.pool):
                self._update_pool()
                self.pool_ptr = 0

            # Update the number of samples collected so far.
            if self.weights is not None:
                n_samples += np.sum(self.weights[users_pooled])
            else:
                n_samples += len(users_pooled)

        return users

    def __call__(
        self,
        data,
        user_clusters,
        n_negatives,
        item_similarity,
        random_state,
        n_samples_expected=None,
    ):
        random_state = get_random_state(random_state)
        if n_samples_expected is None:
            n_samples_expected = data.train_matrix.nnz * n_negatives
        if self.clusters is None:
            self.clusters = user_clusters

        # TODO Should larger clusters contribute more users/samples?

        user_ids = self.draw_samples(n_samples_expected=n_samples_expected)
        if item_similarity:
            pos_items, neg_items, the_users = [], [], []
            for _, user_id in enumerate(user_ids):
                rated = data.train_matrix[user_id].indices
                liked = np.tile(rated, n_negatives)
                pos_items.append(liked)
                the_users.append(np.full_like(liked, fill_value=user_id))

            users = np.concatenate(the_users)[:n_samples_expected]
            likes = np.concatenate(pos_items)[:n_samples_expected]

            dislikes = negative_probs_for_similar_items(
                data.item_similarities.A,  # TODO: cache expensive conversion
                data.train_matrix.A,  # TODO: cache expensive conversion
                users,
                likes,
                random_state.random(len(likes)),
                1.0,  # exponent,
                int(mp.cpu_count() * 0.9 - 1),  # n_threads
            )

            matrix = np.column_stack((users, likes, dislikes))
        else:
            pos_items, neg_items, the_users = [], [], []
            for _, user_id in enumerate(user_ids):
                rated = data.train_matrix[user_id].indices
                liked = np.tile(rated, n_negatives)
                disliked = random_state.choice(
                    a=data.unrated[user_id].indices,
                    size=len(liked) * n_negatives,
                    replace=True,
                )
                pos_items.append(np.tile(liked, n_negatives))
                neg_items.append(disliked)
                the_users.append(np.full_like(disliked, fill_value=user_id))

            matrix = np.column_stack(
                (
                    np.concatenate(the_users),
                    np.concatenate(pos_items),
                    np.concatenate(neg_items),
                )
            )

            matrix = matrix[:n_samples_expected]

        # Shuffling destorys the order of samples in the mini-batch.
        # Therefore, we return the matrix as-is
        assert len(matrix) == n_samples_expected
        return matrix


@dataclass
class TrainConfig:
    decay_mode: str = DECAY_NONE
    epochs: int = DEFAULT_EPOCHS
    get_triplets_fn: Optional[Callable] = None
    intervention: int = 0
    item_similarity: bool = False
    lr: float = DEFAULT_LR
    n_clusters: int = 0
    n_factors: int = DEFAULT_N_FACTORS
    n_negatives: int = DEFAULT_N_NEGATIVES
    propagate: bool = False
    random_state: RandomGenerator = DEFAULT_RANDOM_STATE
    sampler: Optional[str] = SAMPLER_DEFAULT
    samples_per_epoch: Optional[int] = None
    udiv_weights: bool = False
    user_clusters: Optional[UserClusters] = None
    warm_up: int = 0

    def get_samples_per_epoch(self, dataset: Dataset, fraction: float = 0.33):
        n_samples = dataset.train_matrix.nnz
        return int(n_samples * self.n_negatives * fraction)

    @property
    def is_sgd(self):
        return all(
            [
                self.decay_mode == DECAY_NONE,
                self.intervention == 0,
                self.n_clusters == 0,
                self.warm_up == 0,
                self.sampler == SAMPLER_DEFAULT,
            ]
        )

    @property
    def title(self):
        """Description used in W&B"""
        if self.is_sgd:
            return "sgd"
        else:
            n_clusters = self.n_clusters or 0
            intervention = self.intervention or 0
            propagate = str(self.propagate)[0].upper()
            item_similarity = str(self.item_similarity)[0].upper()
            udiv_weights = str(self.udiv_weights)[0].upper()
            warm_up = self.warm_up or 0
            return "-".join(
                [
                    self.sampler,
                    self.decay_mode,
                    f"clu{n_clusters}",
                    f"int{intervention}",
                    f"pro{propagate}",
                    f"wup{warm_up}",
                    f"is{item_similarity}",
                    f"udw{udiv_weights}",
                ]
            )

    def get_wandb_config(self):
        """Returns the hyper-parameters of this experiment for W&B"""
        wandb_config = {
            k: v
            for k, v in asdict(self).items()
            if k not in ["title", "get_triplets_fn", "is_sgd"]
            and (isinstance(v, (str, int, float)) or v is None)
        }
        return wandb_config

    def get_train_kwds(self):
        """Returns parameters to pass to the train method"""
        d = {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in [
                "is_sgd",
                "item_similarity",
                "n_clusters",
                "sampler",
                "udiv_weights",
            ]
        }
        return d


def get_triplets_bpr(
    train_matrix: CSRMatrix,
    n_negatives: int = 5,
    random_state: Optional[RandomGenerator] = None,
    n_samples_expected=None,
) -> np.ndarray:
    random_state = get_random_state(random_state)
    random_state = np.random.default_rng()
    items = np.arange(train_matrix.shape[1])
    buffer = []
    for u in range(train_matrix.shape[0]):
        rated = train_matrix[u].indices
        disliked = random_state.choice(
            a=np.setdiff1d(items, rated),
            size=len(rated) * n_negatives,
            replace=True,
        )
        liked = np.tile(rated.reshape(-1, 1), n_negatives)
        batch = np.concatenate(
            [
                np.full([len(liked) * n_negatives], u).reshape(-1, 1),
                liked.reshape(-1, 1),
                disliked.reshape(-1, 1),
            ],
            axis=1,
        )
        buffer.append(batch)

    matrix = np.concatenate(buffer, axis=0)

    # NOTE Shuffling is important to mimick random sampling
    shuffled_indices = random_state.permutation(np.arange(len(matrix)))
    matrix = matrix[shuffled_indices].astype(np.int32)
    if n_samples_expected < len(matrix):
        print(f"Reducing samples to {n_samples_expected} from {len(matrix)=}")
        matrix = matrix[:n_samples_expected]
    return matrix


def get_triplets_similar_items(
    data: Dataset,
    n_negatives: int = 5,
    exponent: float = 1.0,
    random_state: RandomGenerator = None,
    n_samples_expected=None,
):
    """Creates mini-batches where negative items are similar to their positive counterparts"""
    assert isinstance(data.item_similarities, scipy.sparse.csr_matrix)
    random_state = get_random_state(random_state)
    n_threads = int(mp.cpu_count() * 0.9 - 1)
    user_ids, item_ids = data.train_matrix.nonzero()
    users = np.concatenate([user_ids] * n_negatives).flatten().astype(np.int32)
    likes = np.concatenate([item_ids] * n_negatives).flatten().astype(np.int32)

    dislikes = negative_probs_for_similar_items(
        data.item_similarities.A,  # TODO: cache expensive conversion
        data.train_matrix.A,  # TODO: cache expensive conversion
        users,
        likes,
        random_state.random(len(likes)),
        exponent,  # exponent
        int(mp.cpu_count() * 0.9 - 1),  # n_threads
    )
    matrix = np.column_stack((users, likes, dislikes))
    idxs_shuffled = random_state.permutation(np.arange(matrix.shape[0]))
    matrix = matrix[idxs_shuffled].astype(np.int32)
    if n_samples_expected:
        matrix = matrix[:n_samples_expected]
    return matrix


def train(
    data: Dataset,
    user_clusters: Optional[UserClusters] = None,
    propagate: bool = False,
    get_triplets_fn: Optional[TripletCreator] = None,
    decay_mode: str = "zero",
    epochs: int = 10,
    intervention: int = 0,
    lr: float = 0.01,
    n_factors: int = 10,
    n_negatives: int = 5,
    plot: bool = True,
    reg: float = 0.01,
    random_state: Optional[RandomGenerator] = None,
    samples_per_epoch: Optional[int] = None,
    title: str = "default",
    warm_up: int = 0,
    wlogger=None,
    name="default",
):
    random_state = get_random_state(random_state)

    # validation
    assert decay_mode in DECAY_MODES, f"Unsupported decay function: {decay_mode}"
    assert isinstance(
        user_clusters, UserClusters
    ), f"Expected user_clusters to be of type UserClusters but got {type(user_clusters)}"

    if warm_up > 0:
        assert warm_up + 1 < epochs
        if intervention > 0:
            assert warm_up + 1 < intervention

    if samples_per_epoch is None:
        samples_per_epoch = data.train_matrix.nnz * n_negatives
    assert isinstance(samples_per_epoch, int)

    # for tracking consistency of user predictions
    user_perf_counts = np.zeros(shape=[data.n_users], dtype=np.int32)
    user_perf_totals = np.zeros(shape=[data.n_users], dtype=np.int32)

    # initialise parameters
    u_factors = random_state.normal(0, 0.01, [data.n_users, n_factors])
    i_factors = random_state.normal(0, 0.01, [data.n_items, n_factors])

    precisions, recalls, losses, accuracies = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    mask = data.train_matrix.A > 0
    if get_triplets_fn is None:
        print("Fallback to default triplet creator: %s" % title)
        get_triplets_fn = lambda: get_triplets_bpr(data.train_matrix, n_negatives)

    # Controls whether updates should be shared within-clusters
    share_updates = propagate

    # start training loop
    with tqdm(total=epochs) as progress:
        for epoch in range(epochs):
            # get data
            is_intervention_phase = intervention and epoch > intervention
            is_warm_up_phase = warm_up and epoch < warm_up

            if is_intervention_phase or is_warm_up_phase:
                get_triplets_fn = lambda: get_triplets_bpr(
                    data.train_matrix, n_negatives, n_samples_expected=samples_per_epoch
                )  # type: np.ndarray
                share_updates = False
            else:
                share_updates = propagate
            triplets = get_triplets_fn()
            if samples_per_epoch != len(triplets):
                raise ValueError(
                    f"Expected {samples_per_epoch:,} triplets but got {len(triplets):,} in {title}"
                )
            decay = 1.0  # type: float
            if decay_mode == DECAY_LINEAR:
                decay = 1.0 / (epoch + 1)
            elif decay_mode == DECAY_QUADRATIC:
                decay = 1.0 / (epoch * epoch + 1)
            elif decay_mode == DECAY_EXPONENTIAL:
                decay = 1 / math.exp(epoch + 1)
            elif decay_mode == DECAY_LOGARITHMIC:
                decay = math.log(epoch + 1)

            # update parameters
            loss, accuracy = bpr_update(
                i_factors,
                u_factors,
                triplets[:, 0],
                triplets[:, 1],
                triplets[:, 2],
                user_perf_counts,
                user_perf_totals,
                user_clusters.matrix,
                user_clusters.lookup,
                lr,
                reg,
                share_updates,
                decay,
                NUM_THREADS,
            )
            losses[title].append(loss)
            accuracies[title].append(accuracy)

            # evaluate model
            predictions = u_factors.dot(i_factors.T)
            predictions[mask] = -np.inf
            recommendations = np.argsort(predictions, axis=1)[:, -10:]
            x1 = evaluate(recommendations, data.test_matrix)
            precisions[title].append(x1["PRECISION@10"])
            recalls[title].append(x1["RECALL@10"])

            if wlogger is not None:
                wlogger.log(
                    {
                        "loss": loss,
                        "precision": x1["PRECISION@10"],
                        "recall": x1["RECALL@10"],
                        "accuracy": accuracy,
                        "users": len(np.unique(triplets[:, 0])),
                        "items_pos": len(np.unique(triplets[:, 1])),
                        "items_neg": len(np.unique(triplets[:, 2])),
                        "name": name,
                    }
                )

                progress.update()
                progress.set_postfix(
                    {
                        "Prec@10": "%.4f%%" % (x1["PRECISION@10"]),
                        "Samples": "{:,}".format(len(triplets)),
                        "Interv": "{}".format(intervention),
                        "Decay": "{:.3f}".format(decay),
                        "ALG": title,
                    }
                )

    # render results
    if plot:
        plot_metrics(losses, precisions, title=title, fname=f"fig-{title}.png")
    return losses[title], precisions[title], recalls[title]


cached_clusters = {}  # type: Dict[int, UserClusters]


def get_clusters(data, n_clusters):
    global cached_clusters

    if n_clusters in cached_clusters:
        clustered = cached_clusters[n_clusters]
    else:
        # Default SGD sampler likely to have n_clusters in {0, None},
        # but the clustering requires n_clusters > 1
        n_clusters = DEFAULT_N_CLUSTERS if n_clusters in (0, None) else n_clusters
        clustered = UserClusters.cluster_users(
            train_matrix=data.train_matrix,
            n_clusters=n_clusters,
            random_state=DEFAULT_RANDOM_STATE,
        )
        cached_clusters[n_clusters] = clustered

    return clustered


def train_model(config, data, suffix=None, fraction=1):
    """
    Args:
        config (TrainConfig): training configuration
        data (Dataset): input dataset
        suffix (str, optional): suffix to name of experiment in WandB
        batch_size (float, optional): fraction of data to use per epoch
    """
    global cached_clusters
    config.samples_per_epoch = config.get_samples_per_epoch(data, fraction=fraction)

    if config.user_clusters is None:
        config.user_clusters = get_clusters(data=data, n_clusters=config.n_clusters)
    print(config)

    suffix = suffix or ""
    wandb_name = config.title + suffix
    with wandb_logger(
        name=wandb_name.replace(".", "").strip(),
        wandb_project=WANDB_PROJECT,
        wandb_config=config.get_wandb_config(),
    ) as wlogger:

        result = train(
            data=data,
            wlogger=wlogger,
            plot=False,
            title=config.sampler,
            name=config.title,
            **config.get_train_kwds(),
        )

        # losses[config.title] = result[0]
        # precisions[config.title] = result[1]
        # recalls[config.title] = result[2]

        # sync to wandb
        for k, v in zip(["loss", "precision"], [result[0], result[1]]):
            if k == "loss":
                best_val_idx, best_val = np.argmin(v), np.min(v)
            else:
                best_val_idx, best_val = np.argmax(v), np.max(v)
            wlogger.summary["best_%s" % k] = best_val
            wlogger.summary["best_%s_epoch" % k] = best_val_idx


def train_item_sim(data, fraction=1, warm_up=0, intervention=0, decay_mode=DECAY_NONE):
    """
    Args:
        data (Dataset): input dataset
    """
    config = TrainConfig(
        sampler=SAMPLER_SIMITEM,
        propagate=False,
        n_clusters=0,
        warm_up=warm_up,
        intervention=intervention,
        decay_mode=decay_mode,
    )
    config.get_triplets_fn = lambda: get_triplets_similar_items(
        data=data,
        n_negatives=config.n_negatives,
        random_state=config.random_state,
        n_samples_expected=config.samples_per_epoch,
    )
    train_model(config=config, data=data, fraction=fraction)


def train_user_div(
    data,
    fraction=1,
    intervention=0,
    n_clusters=DEFAULT_N_CLUSTERS,
    propagate=False,
    warm_up=0,
    udiv_weights=False,
):
    print(n_clusters, "n_clusters")
    config = TrainConfig(
        intervention=intervention,
        n_clusters=n_clusters,
        propagate=propagate,
        sampler=SAMPLER_DIVUSER,
        warm_up=warm_up,
        udiv_weights=udiv_weights,
    )
    profile_sizes = [
        len(data.train_matrix[row_idx].indices)
        for row_idx in range(data.train_matrix.shape[0])
    ]
    weights = profile_sizes if config.udiv_weights else None
    config.user_clusters = get_clusters(data=data, n_clusters=n_clusters)
    user_diversity_sampler = UserSampler(
        cluster_sizes=config.user_clusters.cluster_sizes,
        clusters=config.user_clusters.matrix,
        pool_size=100,
        weights=weights,
    )
    config.get_triplets_fn = lambda: user_diversity_sampler(
        data=data,
        user_clusters=config.user_clusters.matrix,
        n_negatives=config.n_negatives,
        random_state=config.random_state,
        item_similarity=config.item_similarity,
        n_samples_expected=config.samples_per_epoch,
    )
    train_model(config=config, data=data, fraction=fraction)


def train_default(data, fraction=1):
    """
    Args:
        data (Dataset): input dataset
    """
    config = TrainConfig(sampler=SAMPLER_DEFAULT)
    config.get_triplets_fn = lambda: get_triplets_bpr(
        train_matrix=data.train_matrix,
        n_negatives=config.n_negatives,
        random_state=config.random_state,
        n_samples_expected=config.samples_per_epoch,
    )
    train_model(config=config, data=data, fraction=fraction)


def main():
    df = load_ml100k("~/datasets/ml-100k/u.data")
    # df = load_tripadvisor("~/datasets/tripadvisor/tripadvisor.csv")

    data = Dataset(df)
    fraction = 1

    # Default
    train_default(data=data, fraction=fraction)

    # User diversity
    # train_user_div(data=data, fraction=fraction)

    # TODO User similarity sampler
    # train_user_sim(data=data, fraction=fraction)

    # Item similarity sampler
    train_item_sim(
        data=data, fraction=fraction, decay_mode=DECAY_EXPONENTIAL, intervention=5
    )


if __name__ == "__main__":
    main()
