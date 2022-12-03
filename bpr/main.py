import logging
import math
import os
import time
from datetime import datetime
from functools import partial
from typing import Callable, List, NamedTuple
from abc import abstractmethod
import click
from click.core import Option
import matplotlib.pyplot as plt
import numpy as np
from click.decorators import option
from scipy.special import softmax
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from typing import Optional, Tuple, Union, Set
from _bpr import (
    bpr_multi_update_batch,
    compute_probabilities,
    resample,
    update_history,
)

# from _bprmb import (
#     bpr_multi_update_batch,
#     compute_probabilities,
#     resample,
#     update_history,
# )

try:
    from .utils import (
        Dataset,
        SPLIT_STRATIFIED,
        asdict,
        calculate_metrics,
        get_rng,
        load_dataset,
        partial_ranking_mask,
        timed_op,
        wandb_logger,
    )
except ImportError:
    # HACK: added to appease linter
    from utils import (
        Dataset,
        SPLIT_STRATIFIED,
        asdict,
        calculate_metrics,
        get_rng,
        load_dataset,
        partial_ranking_mask,
        timed_op,
        wandb_logger,
    )

# Constants
DATASET_ML100K = "ml100k"

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
log = logging.getLogger()


from dataclasses import dataclass, field


@dataclass
class Params:
    """
    Parameters for default SGD
    """

    dataset: Dataset

    I: np.ndarray = field(init=False)
    I_tmp: np.ndarray = field(init=False)
    U: np.ndarray = field(init=False)
    U_tmp: np.ndarray = field(init=False)
    batch_size: int = 32
    diff_limit: int = 0
    epochs: int = 300  # previously 80
    exclude: Set[str] = field(default_factory=set, init=False)
    lr: float = 0.05
    n_factors: int = 10
    n_items: int = 0
    n_negatives: int = 10
    n_users: int = 0
    prob_i: np.ndarray = field(init=False)
    prob_u: np.ndarray = field(init=False)
    reg: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        # Initialise user and item factors
        rng = get_rng(self.seed)
        self.n_users = self.dataset.n_users
        print("\n\n", self.n_users)
        self.n_items = self.dataset.n_items
        self.U = rng.normal(0, 0.01, [self.n_users, self.n_factors])
        self.I = rng.normal(0, 0.01, [self.n_items, self.n_factors])
        self.U_tmp = np.zeros(self.U.shape)
        self.I_tmp = np.zeros(self.I.shape)
        # Initialise selection probabilities
        self.prob_u = np.full([self.n_users], 1)
        self.prob_u = self.prob_u / np.sum(self.prob_u)
        self.prob_i = np.full([self.n_items], 1)
        self.prob_i = self.prob_i / np.sum(self.prob_i)

        # Update variable that shouldn't be synced to https://wandb.ai
        self.exclude = set(
            [
                "I",
                "I_tmp",
                "U",
                "U_tmp",
                "dataset",
                "exclude",
                "n_factors",
                "n_items",
                "n_negatives",
                "n_users",
                "prob_i",
                "prob_u",
                "reg",
                "seed",
            ]
        )

    @abstractmethod
    def pre_epoch_hook(self, epoch: int) -> None:
        pass

    @abstractmethod
    def post_epoch_hook(self, epoch: int) -> None:
        pass

    def __str__(self) -> str:
        reg = str(self.reg).replace(".", "")
        return "baseline-bs{}-rg{}".format(self.batch_size, reg)


@dataclass
class ParamsAB(Params):
    """
    Parameters for Active Bias
    """

    epoch_a: int = field(init=False)
    epoch_z: int = field(init=False)
    fixed_term: float = field(init=False)
    hist_i: np.ndarray = field(init=False)
    hist_i_count: np.ndarray = field(init=False)
    hist_u: np.ndarray = field(init=False)
    hist_u_count: np.ndarray = field(init=False)
    losses: np.ndarray = field(init=False)

    se_0: int = 10
    se_z: Optional[int] = None
    warm_up: int = 0
    window_size: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()

        self.epoch_a = self.warm_up
        self.epoch_z = self.epochs
        assert self.epoch_z - self.epoch_a > 0

        # Compute fixed_term using definition in Equation 9.
        # The previous implementation depends on the se_z values and was based
        # on a particular version of the source code,
        self.fixed_term = math.exp(
            (math.log(1 / self.se_0)) / (self.epoch_z - self.epoch_a)
        )

        # Initialise user and item histories
        self.hist_u = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i = np.zeros([self.n_items, self.epochs], dtype=np.int32)
        self.hist_u_count = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i_count = np.zeros([self.n_items, self.epochs], dtype=np.int32)

        # Update variables that shouldn't be sent to http://wandb.ai
        self.exclude.update(
            [
                "epoch_a",
                "epoch_z",
                "fixed_term",
                "hist_i",
                "hist_i_count",
                "hist_u",
                "hist_u_count",
                "losses",
            ]
        )

    def post_epoch_hook(
        self,
        epoch: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        update_history(
            self.losses,
            self.users,
            self.item_i,
            self.item_j,
            self.hist_u,
            self.hist_i,
            self.hist_u_count,
            self.hist_i_count,
            epoch,
            self.n_users,
            self.n_items,
        )

        if epoch > self.warm_up:
            # Todo: implement selection pressure decay
            # Compute selection pressure
            epoch = max(epoch - self.warm_up, 1)
            se = self.se_0 * math.pow(self.fixed_term, (epoch - self.epoch_a))

            self.prob_u, self.prob_i = compute_probabilities(
                self.hist_u,
                self.hist_i,
                self.hist_u_count,
                self.hist_i_count,
                se,
                self.window_size,
                epoch,  # TODO Confirm if it should be `epochs`
            )

    def __str__(self) -> str:
        prefix = super().__str__().replace("baseline-", "")
        return "ab-{}-ea{}-ez{}-se0{}-sez{}-wu{}-ws".format(
            prefix,
            self.epoch_a,
            self.epoch_z,
            self.se_0,
            self.se_z,
            self.warm_up,
            self.window_size,
        )


@dataclass
class ParamsRB(ParamsAB):
    """
    Parameters for Recency Bias
    """

    def __str__(self) -> str:
        return super().__str__().replace("ab-", "rb-")


def params_to_str(p: Params) -> str:
    """
    Get a descriptive text describing the values of a parameter set

    This is used to label experiments in https://wandb.ai/
    """
    if isinstance(p, Params):
        reg = str(p.reg).replace(".", "")
        return "baseline-bs{}-rg{}".format(p.batch_size, reg)
    else:
        raise ValueError


def evaluate(
    U: np.ndarray,
    I: np.ndarray,
    dataset: Dataset,
    mask: Optional[np.ndarray] = None,
    topk: int = 10,
) -> Tuple[float, float]:
    pred = U.dot(I.T)
    pred[dataset.train_matrix > 0] = -np.inf

    if mask is not None:
        pred[mask == 0] = -np.inf

    rec = np.argsort(pred, axis=1)[:, -topk:]
    x = calculate_metrics(rec, dataset.test_matrix)
    return x["PRECISION@10"], x["RECALL@10"]


def train_bpr(
    params: Union[Params, ParamsAB, ParamsRB],
    mask=None,
    wlogger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ "
    Args:
        item_probs_uniform: whether item probabilies should be updated
    """
    dataset = params.dataset
    n_samples = len(dataset.train_df) * params.n_negatives
    # Generate negative samples
    train_matrix = (dataset.train_matrix > 0).astype(np.int32)
    pos_matrix = train_matrix.copy()
    neg_matrix = (np.logical_not(train_matrix)).astype(np.int32)
    pos_prob_matrix = np.zeros(dataset.train_matrix.shape)
    neg_prob_matrix = np.zeros(dataset.train_matrix.shape)

    # TODO ensure epoch_z > epoch_a
    rng = get_rng(params.seed)

    precisions = np.zeros(params.epochs, dtype=np.float16)
    recalls = np.zeros(params.epochs, dtype=np.float16)
    n_users = dataset.n_users
    n_items = dataset.n_items

    # Initialise user and item factors
    U = rng.normal(0, 0.01, [n_users, params.n_factors])
    I = rng.normal(0, 0.01, [n_items, params.n_factors])

    U_tmp = np.zeros(U.shape)
    I_tmp = np.zeros(I.shape)

    for e in range(params.epochs):
        log.debug("Epoch %0.3d", e)
        if params.pre_epoch_hook:
            params.pre_epoch_hook(epoch=e)

        # resample training data from train rating matrix using user/item
        # probabilities.
        # If biased=False, the probilities will be uniform and not change
        sampled = resample(
            pos_matrix,  # boolean matrix of items rated by users
            neg_matrix,  # boolean matrix of items not rated by users
            pos_prob_matrix,  # probabilities for selecting positive items
            neg_prob_matrix,  # probabilies for selecting negative items
            params.prob_u,  # probabilities for selecting users
            params.prob_i,  # probabilities for selecting items
            n_samples,  # num_samples
            72,  # num_threads
        )

        params.users, params.item_i, params.item_j = sampled

        # resample gets 800784 users
        print("params.users.shape", params.users.shape)

        # update user/item factors
        U, I, loss, params.losses = bpr_multi_update_batch(
            I,
            U,
            I_tmp,
            U_tmp,
            params.users,
            params.item_i,
            params.item_j,
            len(params.users),
            params.batch_size,
            params.lr,
            params.lr,
            params.reg,
            params.diff_limit,
        )

        precision, recall = evaluate(U=U, I=I, dataset=dataset, mask=mask)
        precisions[e] = precision
        recalls[e] = recall

        if wlogger is not None:
            wlogger.log(
                {"loss": loss, "precision": precisions[e], "recall": recalls[e]}
            )

        if params.post_epoch_hook:
            params.post_epoch_hook(epoch=e)
    return precisions, recalls


@click.command()
@click.option("-d", "--dataset", required=True, help="Dataset")
@click.option("-e", "--epochs", default=100, help="Epochs")
@click.option("--quiet", default=False, is_flag=True)
def main(
    dataset,
    epochs,
    quiet=False,
):
    log.setLevel(logging.INFO if quiet else logging.DEBUG)
    t0 = time.time()
    ds = load_dataset(name=dataset, split_type=SPLIT_STRATIFIED)

    # mask should be None if metric isn't Hit Ratio
    # mask = partial_ranking_mask(ds.train_matrix, ds.test_matrix)
    mask = None
    today = datetime.now().strftime("%Y%m%d")[2:]
    wandb_project = f"{today}-{dataset}"
    wandb_project = wandb_project.replace("ml100k", "ml-100k")
    params = [
        Params(dataset=ds, epochs=epochs),
        # ParamsAB(
        #     dataset=ds,
        #     epochs=epochs,
        # ),
        ParamsRB(dataset=ds, epochs=epochs),
        ParamsRB(dataset=ds, epochs=epochs, warm_up=50),
    ]

    for opts in params:
        name = str(opts)
        wandb_config = asdict(opts)
        for x in opts.exclude:
            if x in wandb_config:
                del wandb_config[x]

        with wandb_logger(
            name=name, wandb_project=wandb_project, wandb_config=wandb_config
        ) as wlogger:
            with timed_op(msg=name, logger=log, level=logging.INFO):
                result = train_bpr(
                    mask=mask,
                    params=opts,
                    wlogger=wlogger,
                )
                d = {"precision": result[0], "recall": result[1]}
                for k, v in d.items():
                    best_val_idx, best_val = np.argmax(v), np.max(v)
                    wlogger.summary["best_%s" % k] = best_val
                    wlogger.summary["best_%s_epoch" % k] = best_val_idx

    # TODO consider wloggerining experiment multiple times.

    duration = time.time() - t0
    log.info("Finished experiments in %d seconds", duration)


if __name__ == "__main__":
    main()
