import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import argparse
import math
import os

import numpy as np
import torch as t

try:
    from fdmf.utils.data_processing import load_ml100k
    from fdmf.utils.evaluation import calculate_metrics
except ModuleNotFoundError:
    from utils import calculate_metrics, load_ml100k

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type, Union

from sklearn.metrics.pairwise import cosine_similarity

from _bpr import *
from NCF import NCF

try:
    from .kutils import (
        load_amazon_electronics,
        load_amazon,
        load_gowalla,
        load_lastfm,
        load_mind,
        load_ml1m,
        load_ml100k,
        sample,
        trim,
    )
    from .utils import create_dataset, get_rng, wandb_logger
except ImportError:
    from kutils import (
        load_amazon_electronics,
        load_amazon,
        load_gowalla,
        load_lastfm,
        load_mind,
        load_ml1m,
        load_ml100k,
        sample,
        trim,
    )
    from utils import create_dataset, get_rng, wandb_logger

try:
    from ray import tune
except ImportError:
    tune = object()
    tune.report = lambda x: True
except ModuleNotFoundError:
    tune = object
    tune.report = lambda x: True

rng = get_rng(100)
device = "cuda"

WANDB_PROJECT = "myproject4-bpr"


def inside_tune():
    return tune.session._session is None


class SamplerBase(ABC):
    def __init__(self, epochs=0):
        self.epoch = 0
        self.epochs = epochs

    @abstractmethod
    def sample(self):
        raise NotImplemented

    def pre_epoch(self, epoch):
        self.epoch = epoch

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):
        self.epoch = epoch

    def post_init(self, epochs):
        self.epochs = epochs


class RandomSampler(SamplerBase):
    def __init__(self, data, n_samples, epochs=None):
        super().__init__(epochs=epochs)
        self.data = data.train_rating_matrix
        self.prob_u = np.ones(self.data.shape[0]) / self.data.shape[0]
        self.prob_i = np.ones(self.data.shape[1]) / self.data.shape[1]
        self.prob_j = np.ones(self.data.shape[1]) / self.data.shape[1]
        self.pos_matrix = (self.data > 0).astype(np.int32)
        self.neg_matrix = (self.data == 0).astype(np.int32)
        self.pos_prob_matrix = np.zeros(self.data.shape)
        self.neg_prob_matrix = np.zeros(self.data.shape)
        self.n_samples = n_samples
        self.n_users = data.train_rating_matrix.shape[0]
        self.n_items = data.train_rating_matrix.shape[1]

    def sample(self):
        return resample(
            self.pos_matrix,  # boolean matrix of items rated by users
            self.neg_matrix,  # boolean matrix of items not rated by users
            self.pos_prob_matrix,  # probabilities for selecting positive items
            self.neg_prob_matrix,  # probabilies for selecting negative items
            self.prob_u,  # probabilities for selecting users
            self.prob_i,  # probabilities for selecting items
            self.n_samples,  # num_samples
            72,  # num_threads
        )


class RecencyBiasSampler(RandomSampler):
    def __init__(
        self,
        data,
        n_samples,
        se_0=0.5,  # previously 10,
        se_z=None,
        warm_up=0,
        window_size=2,
        epochs=None,
    ):
        super().__init__(data=data, n_samples=n_samples, epochs=epochs)
        self.se_0 = se_0
        self.se_z = se_z
        self.warm_up = warm_up
        self.window_size = window_size
        self.epoch_a = warm_up

    def post_init(self, epochs):
        self.epochs = epochs
        self.epoch_z = epochs
        self.fixed_term = math.exp(
            (math.log(1 / self.se_0)) / (self.epoch_z - self.epoch_a)
        )
        self.hist_u = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i = np.zeros([self.n_items, self.epochs], dtype=np.int32)
        self.hist_u_count = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i_count = np.zeros([self.n_items, self.epochs], dtype=np.int32)

    def pre_epoch(self, epoch):
        pass

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):
        update_history(
            losses,
            users,
            item_i,
            item_j,
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


class OnlineBatchSampler(RandomSampler):
    def __init__(
        self,
        data,
        n_samples,
        se_0=10,
        se_z=None,
        warm_up=0,
        window_size=10,
    ):
        super().__init__(data, n_samples)
        self.se_0 = se_0
        self.se_z = se_z
        self.warm_up = warm_up
        self.window_size = window_size

    def post_init(self, epochs):
        self.epochs = epochs
        self.epoch_a = self.warm_up
        self.epoch_z = epochs
        self.fixed_term = math.exp(
            (math.log(1 / self.se_0)) / (self.epoch_z - self.epoch_a)
        )

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):

        if epoch > self.warm_up:
            # Todo: implement selection pressure decay
            # Compute selection pressure
            epoch = max(epoch - self.warm_up, 1)
            se = self.se_0 * math.pow(self.fixed_term, (epoch - self.epoch_a))

            self.prob_u, self.prob_i = compute_probabilites_obs(
                users,
                item_i,
                item_j,
                real_losses,
                self.n_users,
                self.n_items,
                se,
                self.n_samples,
            )


class SimilaritySampler(RandomSampler):
    def __init__(self, data, n_samples, exponent=0.1, epochs=100, stop_at=0.5):
        super().__init__(epochs=epochs, data=data, n_samples=n_samples)
        self.data = data.train_rating_matrix
        self.item_similarity = cosine_similarity(
            data.train_rating_matrix.T, data.train_rating_matrix.T
        )
        self.n_samples = n_samples
        self.exponent = exponent
        self.users = data.train_df["user_id"].values.astype(np.int32)
        self.items = data.train_df["item_id"].values.astype(np.int32)
        self.temp_similarity = np.copy(self.item_similarity)
        self.epochs = epochs
        self.stop_at = int(self.epochs * stop_at)

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):

        self.temp_similarity += np.exp(epoch) / np.exp(self.epochs - self.stop_at)
        self.temp_similarity = np.clip(self.temp_similarity, a_min=0, a_max=1)

    def sample(self):
        idx = np.arange(len(self.users))
        idx = np.random.choice(idx, size=self.n_samples, replace=True)
        users = self.users[idx]
        item_i = self.items[idx]
        seeds = np.random.random(len(users))
        item_j = negative_probs_for_similar_items_nodecay(
            self.temp_similarity, self.data, users, item_i, seeds, 72
        )
        return users, item_i, item_j


class RBSimilaritySampler(RandomSampler):
    def __init__(
        self,
        data,
        n_samples,
        exponent=0.5,
        se_0=0.5,  # previously 10,
        se_z=None,
        warm_up=0,
        window_size=2,  # previously 10,
        epochs=100,
        stop_at=0.5,
    ):

        self.data = data.train_rating_matrix
        super().__init__(data, n_samples)
        self.item_similarity = cosine_similarity(data.train_rating_matrix.T)
        self.temp_similarity = np.copy(self.item_similarity)

        self.n_samples = n_samples
        self.exponent = exponent
        self.users = data.train_df["user_id"].values.astype(np.int32)
        self.items = data.train_df["item_id"].values.astype(np.int32)
        self.se_0 = se_0
        self.se_z = se_z
        self.warm_up = warm_up
        self.window_size = window_size
        self.epochs = epochs
        self.epoch_a = warm_up
        self.epoch_z = epochs
        self.fixed_term = math.exp(
            (math.log(1 / self.se_0)) / (self.epoch_z - self.epoch_a)
        )
        self.hist_u = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i = np.zeros([self.n_items, self.epochs], dtype=np.int32)
        self.hist_u_count = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i_count = np.zeros([self.n_items, self.epochs], dtype=np.int32)
        self.stop_at = int(self.epochs * stop_at)

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):
        self.temp_similarity += np.exp(epoch) / np.exp(self.epochs - self.stop_at)
        self.temp_similarity = np.clip(self.temp_similarity, a_min=0, a_max=1)

        update_history(
            losses,
            users,
            item_i,
            item_j,
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

    def sample(self):
        user, item_i, _ = resample(
            self.pos_matrix,  # boolean matrix of items rated by users
            self.neg_matrix,  # boolean matrix of items not rated by users
            self.pos_prob_matrix,  # probabilities for selecting positive items
            self.neg_prob_matrix,  # probabilies for selecting negative items
            self.prob_u,  # probabilities for selecting users
            self.prob_i,  # probabilities for selecting items
            self.n_samples,  # num_samples
            72,  # num_threads
        )

        seeds = np.random.random(len(user))
        item_j = dual_sample_negative(
            self.temp_similarity,
            self.data,
            self.prob_i,
            user,
            item_i,
            seeds,
            72,
        )

        self.exponent = max(0, self.exponent - 2 / self.epochs)
        return user, item_i, item_j


class DualSampler(RandomSampler):
    def __init__(
        self,
        data,
        n_samples,
        exponent=1,
        se_0=0.5,  # previously 10,
        se_z=None,
        warm_up=0,
        window_size=2,
        epochs=100,
    ):

        self.data = data.train_rating_matrix
        super().__init__(data, n_samples)
        self.item_similarity = cosine_similarity(data.train_rating_matrix.T)
        self.n_samples = n_samples
        self.exponent = exponent
        self.users = data.train_df["user_id"].values.astype(np.int32)
        self.items = data.train_df["item_id"].values.astype(np.int32)
        self.se_0 = se_0
        self.se_z = se_z
        self.warm_up = warm_up
        self.window_size = window_size
        self.epochs = epochs

    def post_init(self, epochs):
        self.epochs = epochs
        self.epoch_a = warm_up
        self.epoch_z = epochs
        self.fixed_term = math.exp(
            (math.log(1 / self.se_0)) / (self.epoch_z - self.epoch_a)
        )
        self.hist_u = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i = np.zeros([self.n_items, self.epochs], dtype=np.int32)
        self.hist_u_count = np.zeros([self.n_users, self.epochs], dtype=np.int32)
        self.hist_i_count = np.zeros([self.n_items, self.epochs], dtype=np.int32)

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):
        update_history(
            losses,
            users,
            item_i,
            item_j,
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

    def sample(self):
        user, item_i, _ = resample(
            self.pos_matrix,  # boolean matrix of items rated by users
            self.neg_matrix,  # boolean matrix of items not rated by users
            self.pos_prob_matrix,  # probabilities for selecting positive items
            self.neg_prob_matrix,  # probabilies for selecting negative items
            self.prob_u,  # probabilities for selecting users
            self.prob_i,  # probabilities for selecting items
            self.n_samples,  # num_samples
            72,  # num_threads
        )

        seeds = np.random.random(len(user))
        item_j = negative_probs_for_similar_items(
            self.item_similarity, self.data, user, item_i, seeds, self.exponent, 72
        )

        self.exponent = max(0, self.exponent - 2 / self.epochs)

        return user, item_i, item_j


class DualSampler3(RandomSampler):
    def __init__(
        self,
        data,
        n_samples,
        exponent=0.5,
        se_0=0.5,  # previously 10,
        se_z=None,
        warm_up=0,
        window_size=2,  # previously 10,
        epochs=100,
    ):

        self.data = data.train_rating_matrix
        super().__init__(data, n_samples)
        self.item_similarity = cosine_similarity(data.train_rating_matrix.T)
        self.n_samples = n_samples
        self.exponent = exponent
        self.users = data.train_df["user_id"].values.astype(np.int32)
        self.items = data.train_df["item_id"].values.astype(np.int32)
        self.se_0 = se_0
        self.se_z = se_z
        self.warm_up = warm_up
        self.window_size = window_size
        self.epochs = epochs
        self.epoch_a = warm_up
        self.epoch_z = epochs
        self.fixed_term = math.exp(
            (math.log(1 / self.se_0)) / (self.epoch_z - self.epoch_a)
        )
        self.hist_u = np.zeros([self.n_users], dtype=np.int32)
        self.hist_i = np.zeros([self.n_items], dtype=np.int32)
        self.hist_u_count = np.zeros([self.n_users], dtype=np.int32)
        self.hist_i_count = np.zeros([self.n_items], dtype=np.int32)

    def post_epoch(self, epoch, losses, real_losses, users, item_i, item_j):
        update_history_ex(
            losses,
            users,
            item_i,
            item_j,
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

            self.prob_u, self.prob_i = compute_probabilities_ex(
                self.hist_u,
                self.hist_i,
                self.hist_u_count,
                self.hist_i_count,
                se,
                self.window_size,
                epoch,  # TODO Confirm if it should be `epochs`
            )

    def sample(self):
        user, item_i, _ = resample(
            self.pos_matrix,  # boolean matrix of items rated by users
            self.neg_matrix,  # boolean matrix of items not rated by users
            self.pos_prob_matrix,  # probabilities for selecting positive items
            self.neg_prob_matrix,  # probabilies for selecting negative items
            self.prob_u,  # probabilities for selecting users
            self.prob_i,  # probabilities for selecting items
            self.n_samples,  # num_samples
            72,  # num_threads
        )

        seeds = np.random.random(len(user))
        item_j = dual_sample_negative(
            self.item_similarity,
            self.data,
            self.prob_i,
            user,
            item_i,
            seeds,
            72,
        )

        self.exponent = max(0, self.exponent - 2 / self.epochs)
        return user, item_i, item_j


def _evaluate(pred, dataset, topN=10):
    pred[dataset.train_rating_matrix > 0] = -np.inf
    rec = np.argsort(pred, axis=1)[:, -topN:]
    # res = calculate_metrics(rec, dataset.test_rating_matrix)
    res = compute_metrics(rec, dataset.test_rating_matrix)
    p = res["PRECISION@10"]
    r = res["RECALL@10"]
    n = res["NDCG@10"]
    h = res["HR@10"]
    return p, r, n, h


def evaluate_sgd(P, Q, dataset=None, mask=None, topN=10):
    pred = P.dot(Q.T)
    return _evaluate(pred, dataset, topN=topN)


def eval_ncf(model, dataset, topN=10):
    pred = np.zeros(dataset.train_rating_matrix.shape)
    n_users = dataset.n_users
    n_items = dataset.n_items

    with t.no_grad():
        for _u in range(n_users):
            u = np.full([n_items], _u)
            i = np.arange(n_items)
            p, _ = model.forward(t.tensor(u).to(device), t.tensor(i).to(device))
            pred[_u] = p.cpu().numpy()

    return _evaluate(pred, dataset, topN=topN)


def train_bpr(
    data,
    sampler,
    lr=0.05,
    reg=0.01,
    n_epochs=100,
    batch_size=32,
    n_factors=10,
    topN=10,
    wlogger=None,
    diff_limit=0.5,
):
    sampler.post_init(n_epochs)
    n_users = data.train_rating_matrix.shape[0]
    n_items = data.train_rating_matrix.shape[1]
    P = rng.normal(0, 0.01, [n_users, n_factors])
    Q = rng.normal(0, 0.01, [n_items, n_factors])
    P_tmp = np.zeros(P.shape)
    Q_tmp = np.zeros(Q.shape)
    precisions, recalls, ndcgs, hrs = [], [], [], []
    losses = []

    for e in range(n_epochs):
        sampler.pre_epoch(epoch=e)
        sampled = sampler.sample()
        users, item_i, item_j = sampled

        P, Q, loss, _losses, _real_losses = bpr_multi_update_batch(
            I=Q,
            U=P,
            I_scratch=Q_tmp,
            U_scratch=P_tmp,
            users=users,
            pos_items=item_i,
            neg_items=item_j,
            n_samples=len(users),
            batch_size=batch_size,
            lr_user=lr,
            lr_item=lr,
            reg=reg,
            diff_limit=diff_limit,
        )

        precision, recall, ndcg, hr = evaluate_sgd(
            P=P, Q=Q, dataset=data, mask=None, topN=topN
        )
        # precision, recall = fast_evaluate(
        #     P,
        #     Q,
        #     data.train_rating_matrix.astype(np.int32),
        #     data.test_rating_matrix.astype(np.int32),
        # )
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hrs.append(hr)
        losses.append(loss)

        if not inside_tune():
            tune.report(precision=precision)
            # tune.report(recall=recall)
            # tune.report(ndcg=ndcg)
            # tune.report(hr=hr)

        print(
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | NDCG: {ndcg:.4f} | HR: {hr:.4f} | diff_limit: {diff_limit: .4f}"
        )

        if wlogger is not None:
            wlogger.log(
                {
                    "loss": loss,
                    "precision": precisions[e],
                    "recall": recalls[e],
                    "ndcg": ndcgs[e],
                    "hr": hrs[e],
                }
            )

        sampler.post_epoch(e, _losses, _real_losses, users, item_i, item_j)

    if wlogger is not None:
        wlogger.summary["diff_limit"] = diff_limit

        metrics_dict = {
            "precision": precisions,
            "recall": recalls,
            "hitratio": hrs,
            "ndcg": ndcgs,
        }
        for m, values in metrics_dict.items():
            best_val_epoch = np.argmax(values)
            best_val = np.max(values)
            wlogger.summary["best_%s" % m] = best_val
            wlogger.summary["best_%s_epoch" % m] = best_val_epoch

    return precisions, recalls, losses


def train_ncf(
    data,
    sampler,
    lr=0.01,
    reg=0.01,
    n_epochs=100,
    batch_size=128,
    n_factors=32,
    topN=10,
    n_layers=3,
    dropout=0.0,
    wlogger=None,
):

    sampler.post_init(n_epochs)
    n_users = data.train_rating_matrix.shape[0]
    n_items = data.train_rating_matrix.shape[1]
    model = NCF(n_users, n_items, n_factors, n_layers, dropout, "both", None, None)
    model.to(device)

    def gener(users, item_i, item_j, batch_size):
        i = 0
        while i < len(users):
            nxt = min(i + batch_size, len(users))
            yield users[i:nxt], item_i[i:nxt], item_j[i:nxt]
            i = nxt

    op = t.optim.SGD(model.parameters(), lr=lr)
    losses = []
    precisions = []
    recalls = []

    for e in range(n_epochs):
        sampler.pre_epoch(epoch=e)
        sampled = sampler.sample()
        users, item_i, item_j = sampled
        users_t = t.LongTensor(users).to(device)
        item_i_t = t.LongTensor(item_i).to(device)
        item_j_t = t.LongTensor(item_j).to(device)
        _losses = []
        _real_losses = []
        count = 0
        epoch_loss = 0

        for u, i, j in gener(users_t, item_i_t, item_j_t, batch_size):

            op.zero_grad()
            pu, embeds_u = model.forward(u, i)
            pj, embeds_i = model.forward(u, j)

            _eu = t.cat(embeds_u, dim=1)
            _ei = t.cat(embeds_i, dim=1)
            _e = t.cat([_eu, _ei], dim=1)

            __losses = -(pu - pj).sigmoid().log() + reg * t.norm(_e, dim=1)
            _gr = (pu > pj).detach().cpu().numpy().astype(np.int32)
            loss = __losses.mean()

            loss.backward()
            op.step()
            count += 1
            epoch_loss += loss.detach().cpu().numpy()
            _real_losses.append(__losses.detach().cpu().numpy())
            _losses.append(_gr)

        sampler.post_epoch(
            e,
            np.concatenate(_losses),
            np.concatenate(_real_losses).astype(np.float64),
            users,
            item_i,
            item_j,
        )
        losses.append(epoch_loss / count)
        p, r = eval_ncf(model, data, topN=topN)
        precisions.append(p)
        recalls.append(r)

        print(f"Precision: {p:.4f} | Recall: {r:.4f}")

        if wlogger is not None:
            wlogger.log(
                {
                    "loss": epoch_loss / count,
                    "precision": precisions[e],
                    "recall": recalls[e],
                }
            )

    return precisions, recalls, losses


def load_data(
    dataset,
    split_ratio=0.8,
    min_ratings=10,
    base_dir="..",
    split_type="stratified",
    samples=None,
):
    if samples is not None:
        assert isinstance(samples, (int, float))

    assert dataset in (
        "ml100k",
        "ml1m",
        "mind",
        "amazon",
    ), f"Unsupported dataset: {dataset}"

    rp = lambda p: base_dir + p

    if dataset == "amazon":
        # df = load_amazon_electronics(rp("/data/amazon/amazon-electronics.csv"))
        df = load_amazon(rp("/data/amazon/amazon-new/"))
    elif dataset == "gowalla":
        df = load_gowalla(rp("/data/gowalla"))
    elif dataset == "lastfm":
        df = load_lastfm(rp("/data/lastfm"))
    elif dataset == "mind":
        df = load_mind(rp("/data/mind/behaviors.tsv"))
    elif dataset == "ml100k":
        df = load_ml100k(rp("/data/ml-100k/u.data"))
    elif dataset == "ml1m":
        df = load_ml1m(rp("/data/ml-1m/ratings.dat"))

    df_trimed = trim(df, copy=True)

    if samples is not None:
        df_trimed = sample(df=df_trimed, size=samples, min_size=min_ratings)

    return create_dataset(
        df_trimed,
        split_ratio=split_ratio,
        min_ratings=min_ratings,
        split_type=split_type,
    )


def execute_now(name, dataset, data, n_negatives, sampler, kwds):
    print("In execute now")
    wandb_config = {"sampler": name}
    with wandb_logger(
        name="{}-bpr-{}".format(dataset, name),
        wandb_project="myproject4-bpr",
        wandb_config=wandb_config,
    ) as wlogger:
        sampler = sampler(data, n_negatives * len(data.train_df))
        train_bpr(data, sampler, wlogger=wlogger, **kwds)


MULTIPROC = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Options include ml100k, ml1m, mind, amazon")
    args = parser.parse_args()

    n_negatives = 10

    data_config = {
        "amazon": dict(samples=1000),
        "ml100k": dict(samples=None),
        "ml1m": dict(samples=None),
        "mind": dict(samples=5000),
        "lastfm": dict(samples=None),
        "gowalla": dict(samples=None),
    }

    data = load_data(args.dataset, samples=data_config[args.dataset]["samples"])

    samplers = {
        "random": RandomSampler,
        "rbsim": RBSimilaritySampler,
        "sim": SimilaritySampler,
        "rb": RecencyBiasSampler,
        "dual3": DualSampler3,
        "obs": OnlineBatchSampler,
    }  # type: Dict[str, Any]

    hparams = {
        "amazon": {
            "random": dict(
                n_epochs=100,
                n_factors=10,
            )
        },
        "mind": dict(n_epochs=200, n_factors=15, lr=0.05, samplers=None),
        "ml100k": dict(n_epochs=500, n_factors=10, samplers=None),
        "lastfm": dict(n_epochs=300, n_factors=10, samplers=None),
        "ml1m": dict(n_epochs=500, n_factors=10, samplers=None),
    }

    if MULTIPROC:
        params, processes = [], []
        for name, sampler in samplers.items():
            kwds = {}
            source = hparams.get(args.dataset, None)
            if source is not None:
                for k, v in source.items():
                    if v is not None and v not in ("samplers",):
                        kwds[k] = v
            print("kwds", kwds)
            params.append((name, args.dataset, data, n_negatives, sampler, kwds))
            p = mp.Process(
                target=execute_now,
                args=(name, args.dataset, data, n_negatives, sampler, kwds),
            )
            p.start()
            processes.append(p)
            # executor.submit(
            #     execute_now, name, args.dataset, data, n_negatives, sampler, kwds
            # )
        for p in processes:
            p.join()
    else:
        # for name, sampler in samplers.items():
        #     kwds = hparams[args.dataset][name]
        #     execute_now(
        #         name=name,
        #         dataset=args.dataset,
        #         sampler=sampler,
        #         data=data,
        #         n_negatives=n_negatives,
        #         kwds=kwds,
        #     )
        pass
