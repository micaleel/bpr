# cython: language_level=3
# cython: profile=True
# cython: linetrace=True

# from libcpp cimport bool
import cython
import os
import random

from cython cimport floating, integral
from cython import boundscheck
from cython.parallel import prange, parallel, threadid
import math
import numpy as np
cimport numpy as np

from libc.math cimport exp, log, log2
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free

cdef extern from "time.h" nogil:
    long int time(int)

cdef extern from "stdlib.h" nogil:
    double drand48()
    void srand48(long int seedval)

cdef extern from "argsort.c" nogil:
    ctypedef struct my_index:
        double value
        int idx
    void argsort_row(double * A, int * out,my_index * mm, int K)
    int binary_search_row(double * arr, int * PP, int n, double target)

cdef extern from "stdlib.h" nogil:
    double drand48()
    void srand48(long int seedval)

cdef extern from "time.h" nogil:
    long int time(int)


def compute_probabilities_ex1(
    integral[:] u_history not None,
    integral[:] i_history not None,
    integral[:] u_count,
    integral[:] i_count,
    double selection_pressure,
    int q,
    int step):

    cdef np.ndarray[np.float64_t, ndim=1] u_out = np.zeros(u_history.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] i_out = np.zeros(i_history.shape[0], dtype=np.float64)

    cdef int user_idx, item_idx
    cdef int i, j, k, pos, n_elements
    cdef double numerator
    cdef double divisor

    with nogil, boundscheck(False):
        for user_idx in range(u_history.shape[0]):
            n_elements = 0
            numerator = 0
            divisor = 0

            if u_count[user_idx] > 0:
                numerator += u_history[user_idx]
                divisor += u_count[user_idx]
                n_elements += 1 

            if divisor == 0:
                divisor = 1

            u_out[user_idx] = 1.0 - (numerator / divisor)
            # u_out[user_idx] = 1/((exp(log(selection_pressure/u_history.shape[0])))**u_out[user_idx])
            u_out[user_idx] = 1/((exp(log(selection_pressure)/u_history.shape[0]))**u_out[user_idx])

        for item_idx in range(i_history.shape[0]):
            n_elements = 0
            numerator = 0
            divisor = 0
            if i_count[item_idx] > 0:
                numerator += i_history[item_idx]
                divisor += i_count[item_idx] 
                n_elements +=1
            if divisor == 0:
                divisor = 1
            i_out[item_idx] = 1.0 - (numerator / divisor)
            i_out[item_idx] = 1/((exp(log(selection_pressure)/i_history.shape[0]))**i_out[item_idx])

    return u_out / u_out.sum(), i_out / i_out.sum()

def compute_probabilities_ex(
    integral[:] u_history not None,
    integral[:] i_history not None,
    integral[:] u_count,
    integral[:] i_count,
    double selection_pressure,
    int q,
    int step):

    cdef np.ndarray[np.float64_t, ndim=1] u_out = np.zeros(u_history.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] i_out = np.zeros(i_history.shape[0], dtype=np.float64)

    cdef int user_idx, item_idx
    cdef int i, j, k, pos, n_elements
    cdef double numerator
    cdef double divisor

    with nogil, boundscheck(False):
        for user_idx in range(u_history.shape[0]):
            n_elements = 0
            numerator = 0
            divisor = 0

            if u_count[user_idx] > 0:
                numerator += u_history[user_idx]
                divisor += u_count[user_idx]
                n_elements += 1 

            if divisor == 0:
                divisor = 1

            u_out[user_idx] = 1.0 - (numerator / divisor)
            # u_out[user_idx] = 1/((exp(log(selection_pressure/u_history.shape[0])))**u_out[user_idx])
            u_out[user_idx] = 1/((exp(log(selection_pressure)/u_history.shape[0]))**u_out[user_idx])

        for item_idx in range(i_history.shape[0]):
            n_elements = 0
            numerator = 0
            divisor = 0
            if i_count[item_idx] > 0:
                numerator += i_history[item_idx]
                divisor += i_count[item_idx] 
                n_elements +=1
            if divisor == 0:
                divisor = 1
            i_out[item_idx] = 1.0 - (numerator / divisor)
            i_out[item_idx] = 1/((exp(log(selection_pressure)/i_history.shape[0]))**i_out[item_idx])

    return u_out / u_out.sum(), i_out / i_out.sum()

def compute_probabilities(
    integral[:, :] u_history not None,
    integral[:, :] i_history not None,
    integral[:, :] u_count,
    integral[:, :] i_count,
    double selection_pressure,
    int q,
    int step):

    cdef np.ndarray[np.float64_t, ndim=1] u_out = np.zeros(u_history.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] i_out = np.zeros(i_history.shape[0], dtype=np.float64)

    cdef int user_idx, item_idx
    cdef int i, j, k, pos, n_elements
    cdef double numerator
    cdef double divisor

    with nogil, boundscheck(False):
        for user_idx in range(u_history.shape[0]):
            n_elements = 0
            numerator = 0
            divisor = 0
            j = step

            # Consider q most-recent history elements
            while j > -1 and n_elements < q:
                if u_count[user_idx, j] > 0:
                    numerator += u_history[user_idx, j]
                    divisor += u_count[user_idx, j]
                    n_elements += 1 
                j -= 1
            if divisor == 0:
                divisor = 1
            u_out[user_idx] = 1.0 - (numerator / divisor)
            u_out[user_idx] = 1/((exp(log(selection_pressure/u_history.shape[0])))**u_out[user_idx])

        for item_idx in range(i_history.shape[0]):
            n_elements = 0
            numerator = 0
            divisor = 0
            j=step
            while j > -1 and n_elements < q:
                if i_count[item_idx, j] > 0:
                    numerator += i_history[item_idx, j]
                    divisor += i_count[item_idx, j] 
                    n_elements +=1
                j-=1
            if divisor == 0:
                divisor = 1
            i_out[item_idx] = 1.0 - (numerator / divisor)
            i_out[item_idx] = 1/((exp(log(selection_pressure/i_history.shape[0])))**i_out[item_idx])

    return u_out / u_out.sum(), i_out / i_out.sum()

def resample(
    np.ndarray[np.int32_t, ndim=2] pos_matrix,
    np.ndarray[np.int32_t, ndim=2] neg_matrix,
    np.ndarray[np.float64_t,ndim=2] pos_prob_matrix,
    np.ndarray[np.float64_t,ndim=2] neg_prob_matrix,
    np.ndarray[np.float64_t, ndim=1] prob_user,
    np.ndarray[np.float64_t, ndim=1] p_item,
    int n_samples,
    int num_threads=72
):
    pos_prob_matrix[:] = pos_matrix * p_item
    pos_prob_matrix /= pos_prob_matrix.sum(axis=1)[:,None]
    neg_prob_matrix[:] = neg_matrix * p_item
    neg_prob_matrix /= neg_prob_matrix.sum(axis=1)[:,None]

    prob_user=np.cumsum(prob_user)
    np.cumsum(pos_prob_matrix,axis=1, out=pos_prob_matrix)
    np.cumsum(neg_prob_matrix, axis=1, out=neg_prob_matrix)

    cdef int thread, i, tmp_item_id, tmp_user_id, n_users, n_items
    n_items = neg_prob_matrix.shape[1]
    n_users = prob_user.shape[0]

    cdef int samples_per_thread = int(n_samples/num_threads)
    cdef np.ndarray[ndim=2, dtype=np.int32_t] users, item_i, item_j
    cdef np.ndarray[ndim=3, dtype=np.float64_t] random_probs
    users = np.zeros([num_threads, samples_per_thread], dtype=np.int32) 
    item_i = np.zeros([num_threads, samples_per_thread], dtype=np.int32) 
    item_j = np.zeros([num_threads, samples_per_thread], dtype=np.int32) 
    random_probs = np.random.random([num_threads, samples_per_thread, 3])


    cdef double r
    cdef double *neg_matrix_ptr = <double *> neg_prob_matrix.data;
    cdef double *pos_matrix_ptr = <double *> pos_prob_matrix.data;
    cdef int *neg_train_matrix_ptr = <int *> neg_matrix.data;
    cdef int *pos_train_matrix_ptr = <int *> pos_matrix.data;

    cdef np.ndarray[ndim=1, dtype=np.int32_t] full_user = np.ones(prob_user.shape[0],dtype=np.int32)
    cdef int *ones_ptr = <int *> full_user.data;
    cdef np.ndarray[ndim=1, dtype=np.float64_t] s = np.zeros(num_threads)
    cdef double * user_prob_ptr = <double *> prob_user.data;

    with nogil, boundscheck(False):
        for thread in prange(num_threads, num_threads=num_threads):
            srand48(42)
            for i in range(samples_per_thread):                
                r = random_probs[thread, i, 0]
                tmp_user_id = binary_search_row(user_prob_ptr, ones_ptr, n_users, r)
                users[thread, i] = tmp_user_id 

                r = random_probs[thread, i, 1]
                tmp_item_id = binary_search_row(
                    pos_matrix_ptr + (tmp_user_id * n_items), 
                    pos_train_matrix_ptr + (tmp_user_id * n_items), 
                    n_items, 
                    r
                )
                item_i[thread, i] = tmp_item_id
                
                r = random_probs[thread, i, 2]
                tmp_item_id = binary_search_row(
                    neg_matrix_ptr + (tmp_user_id * n_items),
                    neg_train_matrix_ptr + (tmp_user_id * n_items), 
                    n_items, 
                    r
                )
                item_j[thread, i] = tmp_item_id

    return users.flatten(), item_i.flatten(), item_j.flatten()

@cython.boundscheck(False)
@cython.cdivision(True) 
@cython.nonecheck(False)
@cython.wraparound(False)
def compute_metrics(
    np.ndarray[np.int64_t, ndim=2] recommendations, 
    np.ndarray[np.float64_t, ndim=2] actual_rating_matrix, 
    bint ndcg=False):

    cdef int max_hits = np.sum(actual_rating_matrix > 0)
    cdef int rateable = np.count_nonzero(np.sum(actual_rating_matrix, axis=1))
    cdef int top_n = recommendations.shape[1]
    cdef int tpfn = rateable * top_n
    cdef float n_tp = 0
    cdef int user, item, r_idx, rank_idx
    cdef int n_users = len(recommendations)
    cdef float ndcgs = 0
    cdef float dcgs = 0, idcgs = 0
    cdef float n_tp_user, hr, hits = 0
    cdef int i, hit = 0

    with nogil:
        for user in range(n_users):
            dcgs = 0
            idcgs = 0
            n_tp_user = 0
            hit = 0
            for rank_idx in range(top_n):
                item = recommendations[user, rank_idx]
                if actual_rating_matrix[user, item] > 0:
                    n_tp_user += 1
                    dcgs += 1 / log2(rank_idx + 2)
                    idcgs += 1 / log2(n_tp_user + 2)
                    hit |= 1
                    
            if idcgs > 0:
                ndcgs += (dcgs / idcgs)
            n_tp += n_tp_user
            hits += hit

    precision = n_tp / tpfn
    recall = n_tp / max_hits
    ndcg_avg = ndcgs / n_users
    hr = hits / n_users

    result = {
        "PRECISION@{}".format(top_n): precision,
        "RECALL@{}".format(top_n): recall,
        "NDCG@{}".format(top_n): ndcg_avg,
        "HR@{}".format(top_n): hits / n_users
    }
    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def bpr_update(
    floating [:, :] I not None,
    floating [:, :] U not None,
    integral [:] users not None,
    integral [:] positives not None,
    integral [:] negatives not None,
    integral [:] user_perf_counts not None,
    integral [:] user_perf_totals not None,
    integral [:, :] cluster_matrix not None,
    integral [:] cluster_lookup not None,
    floating lr,
    floating reg,
    integral propagate = 1,
    floating decay = 0.5,
    int num_threads=10,
 ):

    cdef floating * disliked 
    cdef floating * liked
    cdef floating * user
    cdef floating * cluster_user

    cdef floating loss = 0
    cdef floating score = 0
    cdef floating temp = 0
    cdef floating threshold = 0.0
    cdef floating z = 0

    cdef integral cf_idx = 0
    cdef integral cluster_idx = 0
    cdef integral cluster_matrix_max_size = cluster_matrix.shape[1]
    cdef integral correct = 0
    cdef integral f_idx
    cdef integral idx 
    cdef integral max_average = 0
    cdef integral n_factors = I.shape[1]
    cdef integral n_samples = users.shape[0]
    cdef integral n_users = U.shape[0]
    cdef integral n_clusters = cluster_matrix.shape[0]
    cdef integral u_idx, thread_id

    cdef integral[:] averages = np.zeros(shape=[n_users], dtype=np.int32)
    cdef integral[:] clusters_updated = np.zeros(shape=[n_clusters], dtype=np.int32)
    cdef floating [:] cluster_factors_counts = np.zeros(shape=[n_clusters, ], dtype=np.float64)
    cdef floating [:, :] cluster_factors = np.zeros(shape=[n_clusters, n_factors], dtype=np.float64)
    print('bpr_update')
    with nogil:
        for idx in range(n_samples):
            user = &U[users[idx], 0]
            liked, disliked = &I[positives[idx], 0], &I[negatives[idx], 0]

            # compute difference in prediction (is pred[+ve] > pred[-ve]?) 
            score = 0
            for f_idx in range(n_factors):
                score += user[f_idx] * (liked[f_idx] - disliked[f_idx])

            z = 1 / (1 + exp(score))
            loss += -log(1/(1 + exp(-score)))

            if z < 0.5:
                correct += 1
                user_perf_totals[users[idx]] += 1

            user_perf_counts[users[idx]] += 1

            # update factors
            for f_idx in range(n_factors):
                temp = user[f_idx]
                user[f_idx] += lr * (z * (liked[f_idx] - disliked[f_idx]) - reg * user[f_idx] )
                liked[f_idx] += lr * (z * temp - reg * liked[f_idx])
                disliked[f_idx] += lr * (-z * temp - reg * disliked[f_idx])

        # """"
        if propagate > 0:
            # compute uncertainty
            for u_idx in range(n_users):

                if user_perf_counts[u_idx] != 0:
                    # Implies that user u_idx hasn't been seen during this update
                    averages[u_idx] = user_perf_totals[u_idx] / user_perf_counts[u_idx]

                max_average = max(max_average, averages[u_idx])

            # anything above 40% of the maximum average is stable-ish
            threshold = 0.6 * max_average

            # accumulate factors of "stable" users in the same clusters
            for u_idx in prange(n_users):
                cluster_idx = cluster_lookup[u_idx] 
                cluster_factors_counts[cluster_idx] += 1
                clusters_updated[cluster_idx] = 1
                for cf_idx in range(n_factors):
                    cluster_factors[cluster_idx, cf_idx] += user[cf_idx]
                # if user_perf_totals[u_idx] > threshold:
                #     cluster_idx = cluster_lookup[u_idx] 
                #     cluster_factors_counts[cluster_idx] += 1
                #     clusters_updated[cluster_idx] = 1
                #     for cf_idx in range(n_factors):
                #         cluster_factors[cluster_idx, cf_idx] += user[cf_idx]

            for cluster_idx in range(n_clusters):
                # compute average of accumulated factors
                for cf_idx in range(n_factors):
                    cluster_factors[cluster_idx, cf_idx] = cluster_factors[cluster_idx, cf_idx] /  cluster_factors_counts[cluster_idx]

                # share cluster factors between users of the same cluster
                for cu_idx in range(cluster_matrix_max_size):
                    if cluster_matrix[cluster_idx, cu_idx] != -1:
                        if clusters_updated[cluster_idx] > 0:
                            for cf_idx in range(n_factors):
                                U[cu_idx, cf_idx] = U[cu_idx, cf_idx] + decay * cluster_factors[cluster_idx, cf_idx]
                                U[cu_idx, cf_idx] /= 2

            # reset flags
            for u_idx in range(n_users):
                averages[u_idx] = 0
                # reset perf counters
                user_perf_totals[u_idx] = 0
                user_perf_counts[u_idx] = 0
                
            # reset accumulators
            for cluster_idx in range(n_clusters):
                clusters_updated[cluster_idx] = 0
        # """

    return loss/n_samples, 1.0*correct/n_samples

@cython.cdivision(True)
@cython.boundscheck(False)
def negative_probs_for_similar_items(
    floating[:, :] item_similarity,
    floating[:, :] train_matrix,
    integral[:] user_ids,
    integral[:] item_i,
    
    floating[:] random_seeds,
    double exponent,
    int num_threads=10,
):
    """Pairs each positive item with a 'similar' negative item"""
    cdef float total,r,m

    cdef np.ndarray[np.int32_t, ndim=1] item_j = np.zeros(len(item_i), dtype=np.int32)
    cdef floating[:] cum_probs =np.zeros(len(item_i), dtype=np.float64)

    # Accumulates similiarity values on a per-item basis
    cdef floating[:] totals = np.zeros(len(item_i), dtype=np.float64)
    cdef floating[:] base = np.zeros(len(item_i), dtype=np.float64)
    cdef int i, u, j,idx

    srand48(time(0))
    
    with nogil:
        for idx in prange(item_i.shape[0], num_threads=num_threads):
            u = user_ids[idx]
            i = item_i[idx]
           
            # Accumulate similarity values for negative items 
            for j in range(item_similarity.shape[1]):
                if train_matrix[u, j] == 0.0:
                    totals[idx] += item_similarity[i,j] ** exponent
                    base[idx] += 1.0

            #some items have no similar negative samples
            #here we need to default to something so we just select a random item
            if totals[idx] == 0.0:
                cum_probs[idx] = 0.0
                r = random_seeds[idx]
                for j in range(item_similarity.shape[1]):
                    if train_matrix[u, j] == 0:
                        cum_probs[idx] += 1/base[idx]

                        if cum_probs[idx] >= r:
                            item_j[idx] = j
                            break
            else:
                cum_probs[idx] = 0.0
                r = random_seeds[idx]
                for j in range(item_similarity.shape[1]):
                    if train_matrix[u, j] == 0:
                        # Accumulate probabilities of unrated/negative items
                        cum_probs[idx] += (item_similarity[i, j] ** exponent) / totals[idx]

                        # Select item with cumulative probability above r
                        if cum_probs[idx] >= r:
                            item_j[idx] = j
                            break
    return item_j

@cython.cdivision(True)
@cython.boundscheck(False)
def negative_probs_for_similar_items_nodecay(
    floating[:, :] item_similarity,
    floating[:, :] train_matrix,
    integral[:] user_ids,
    integral[:] item_i,
    
    floating[:] random_seeds,
    int num_threads=10,
):
    """Pairs each positive item with a 'similar' negative item"""
    cdef float total,r,m

    cdef np.ndarray[np.int32_t, ndim=1] item_j = np.zeros(len(item_i), dtype=np.int32)
    cdef floating[:] cum_probs =np.zeros(len(item_i), dtype=np.float64)

    # Accumulates similiarity values on a per-item basis
    cdef floating[:] totals = np.zeros(len(item_i), dtype=np.float64)
    cdef floating[:] base = np.zeros(len(item_i), dtype=np.float64)
    cdef int i, u, j,idx

    srand48(time(0))
    
    with nogil:
        for idx in prange(item_i.shape[0], num_threads=num_threads):
            u = user_ids[idx]
            i = item_i[idx]
           
            # Accumulate similarity values for negative items 
            for j in range(item_similarity.shape[1]):
                if train_matrix[u, j] == 0.0:
                    totals[idx] += item_similarity[i,j] 
                    base[idx] += 1.0

            #some items have no similar negative samples
            #here we need to default to something so we just select a random item
            if totals[idx] == 0.0:
                cum_probs[idx] = 0.0
                r = random_seeds[idx]
                for j in range(item_similarity.shape[1]):
                    if train_matrix[u, j] == 0:
                        cum_probs[idx] += 1/base[idx]

                        if cum_probs[idx] >= r:
                            item_j[idx] = j
                            break
            else:
                cum_probs[idx] = 0.0
                r = random_seeds[idx]
                for j in range(item_similarity.shape[1]):
                    if train_matrix[u, j] == 0:
                        # Accumulate probabilities of unrated/negative items
                        cum_probs[idx] += item_similarity[i, j] / totals[idx]

                        # Select item with cumulative probability above r
                        if cum_probs[idx] >= r:
                            item_j[idx] = j
                            break
    return item_j





@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
def popularity_to_probs(
    floating[:, :] train_matrix,
    int num_threads = 10
):
    """Convert popularity scores to probabilities"""
    cdef int eid = 0
    cdef long n_entities = train_matrix.shape[0]
    cdef double popularity = 0.0
    cdef double total = 0.0
    cdef floating[:] probs = np.zeros(train_matrix.shape[0], dtype=np.float64)

    # Record interaction frequencies
    for eid in range(n_entities):
        popularity = log(train_matrix[eid].nnz)
        probs[eid] = popularity
        total += popularity

    with nogil:
        # Convert frequencies to probabilities
        for eid in prange(n_entities, num_threads=num_threads):
            probs[eid] /= total

    return probs

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def dual_sample_negative(
    floating[:, :] item_similarity,
    floating[:, :] train_matrix,
    floating[:] p_item,
    integral[:] user_ids,
    integral[:] item_i,
    floating[:] random_seeds,
    int num_threads=10,
):
    """Pairs each positive item with a 'similar' negative item"""
    cdef float total,r,m

    cdef np.ndarray[np.int32_t, ndim=1] item_j = np.zeros(len(item_i), dtype=np.int32)
    cdef floating[:] cum_probs =np.zeros(len(item_i), dtype=np.float64)

    # Accumulates similiarity values on a per-item basis
    cdef floating[:] totals = np.zeros(len(item_i), dtype=np.float64)
    cdef floating[:] base = np.zeros(len(item_i), dtype=np.float64)
    cdef int i, u, j,idx
    cdef float combined = 0

    srand48(time(0))
    
    with nogil:
        for idx in prange(item_i.shape[0], num_threads=num_threads):
            u = user_ids[idx]
            i = item_i[idx]
           
            # Accumulate similarity values for negative items 
            for j in range(item_similarity.shape[1]):
                if train_matrix[u, j] == 0.0:
                    combined = item_similarity[i,j] * p_item[j]
                    if combined > 1:
                        combined = 1
                    totals[idx] += combined
                    base[idx] += 1.0

            #some items have no similar negative samples
            #here we need to default to something so we just select a random item
            if totals[idx] == 0.0:
                cum_probs[idx] = 0.0
                r = random_seeds[idx]
                for j in range(item_similarity.shape[1]):
                    if train_matrix[u, j] == 0:
                        cum_probs[idx] += 1/base[idx]

                        if cum_probs[idx] >= r:
                            item_j[idx] = j
                            break
            else:
                cum_probs[idx] = 0.0
                r = random_seeds[idx]
                for j in range(item_similarity.shape[1]):
                    if train_matrix[u, j] == 0:
                        # Accumulate probabilities of unrated/negative items
                        combined = item_similarity[i,j] * p_item[j]
                        if combined > 1:
                            combined = 1
                        cum_probs[idx] += combined / totals[idx]

                        # Select item with cumulative probability above r
                        if cum_probs[idx] >= r:
                            item_j[idx] = j
                            break
    return item_j

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def bpr_multi_update_batch(
    np.ndarray[np.float64_t, ndim=2] I not None,
    np.ndarray[np.float64_t, ndim=2] U not None,
    np.ndarray[np.float64_t, ndim=2] I_scratch not None,
    np.ndarray[np.float64_t, ndim=2] U_scratch not None,
    np.ndarray[np.int32_t, ndim=1] users not None,
    np.ndarray[np.int32_t, ndim=1] pos_items not None,
    np.ndarray[np.int32_t, ndim=1] neg_items not None,
    np.int32_t n_samples,
    np.int32_t batch_size,
    np.float32_t lr_user,
    np.float32_t lr_item,
    np.float32_t reg,
    np.float32_t diff_limit
):
    cdef Py_ssize_t user_id, i, j, u, idx, jdx, n_factors 
    cdef Py_ssize_t batch_idx, prev_batch_idx, kdx
    cdef double cur_batch_size, cur_loss
    cdef double denominator
    cdef double score
    cdef double grad_i, grad_j, grad_u
    cdef double i_factor, j_factor
    cdef double loss
    cdef double neg_predict
    cdef double noise_i, noise_j, noise_u
    cdef double sigmoid, z
    cdef double u_factor 
    cdef float momentum
    cdef np.ndarray[ndim=1,dtype=np.float64_t] U_updates = np.zeros(len(U), dtype=np.float64)
    cdef np.ndarray[ndim=1,dtype=np.float64_t] I_updates = np.zeros(len(I), dtype=np.float64)
    cdef np.ndarray[ndim=1,dtype=np.int32_t] losses = np.zeros(n_samples, dtype=np.int32)
    cdef np.ndarray[ndim=1,dtype=np.int32_t] losses_users = np.zeros(n_samples, dtype=np.int32)
    cdef np.ndarray[ndim=1,dtype=np.int32_t] losses_items = np.zeros(n_samples, dtype=np.int32)
    cdef np.ndarray[ndim=1,dtype=np.float64_t] real_losses = np.zeros(n_samples, dtype=np.float64)

    n_factors = I.shape[1]

    loss = 0
    batch_idx = batch_size
    prev_batch_idx = 0
    for idx in range(0, n_samples):
        u = users[idx]
        i = pos_items[idx]
        j = neg_items[ idx]

        U_updates[u] += 1
        I_updates[i] += 1
        I_updates[j] += 1

        score = 0
        for jdx in range(n_factors):
            score += U[u, jdx] *(I[i, jdx] - I[j, jdx])

        try:
            z = 1 / (1 + exp(score))
            loss += -log(1 / (1 + exp(-score)))
            real_losses[idx] = -log(1 / (1 + exp(-score)))
        except ZeroDivisionError:
            print('z', z, 'exp(-z)', exp(-z))
            print('U[u]', U[u], 'I[i]', I[i], 'I[j]', I[j])
       
        if z < diff_limit:
            # TODO Rename losses to predictions
            losses[idx] = 1
            losses_users[u] += 1
            losses_items[i] += 1
            losses_items[j] += 1

        for jdx in range(n_factors):
            u_factor = U[u, jdx] + 0
            i_factor = I[i, jdx] + 0
            j_factor = I[j, jdx] + 0
            U_scratch[u, jdx] += lr_user * (z * (i_factor - j_factor) - (reg * u_factor))
            I_scratch[i, jdx] += lr_item * (z *  u_factor - (reg * i_factor ))
            I_scratch[j, jdx] += lr_item *  ( (z * (- u_factor)) - (reg * j_factor ))
            
        if idx == n_samples - 1 or idx == batch_idx:
            cur_batch_size = min(batch_size, n_samples-prev_batch_idx)
            for u in range(U_updates.shape[0]):
                if U_updates[u] > 0:
                    for jdx in range(n_factors):
                        U[u, jdx] += (U_scratch[ u, jdx] / cur_batch_size)
                        U_scratch[ u, jdx] = 0
                    U_updates[ u] = 0
            for i in range(I_updates.shape[0]):
                if I_updates[ i] > 0:
                    for jdx in range(n_factors):
                        I[ i, jdx] += (I_scratch[ i, jdx] / cur_batch_size)
                        I_scratch[i, jdx] = 0
                    I_updates[ i] = 0
            prev_batch_idx = batch_idx
            batch_idx = prev_batch_idx + batch_size
        loss = loss / n_samples
    return U, I, loss, losses, real_losses


def compute_probabilites_obs(
    np.ndarray[ndim=1, dtype=np.int32_t] users,
    np.ndarray[ndim=1, dtype=np.int32_t] item_i,
    np.ndarray[ndim=1, dtype=np.int32_t] item_j,
    np.ndarray[ndim=1, dtype=np.float64_t] losses,
    int n_users,
    int n_items,
    float se,
    int n_samples,
):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] loss_u = np.zeros(n_users)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] loss_i = np.zeros(n_items)
    cdef np.ndarray[ndim=1, dtype=np.int32_t] count_u = np.zeros(n_users,dtype=np.int32)
    cdef np.ndarray[ndim=1, dtype=np.int32_t] count_i = np.zeros(n_items,dtype=np.int32)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] prob_u = np.zeros(n_users)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] prob_i = np.zeros(n_items)

    cdef int idx, u, i, j, rank

    for idx in range(losses.shape[0]):
        loss_u[users[idx]] += losses[idx]
        loss_i[item_i[idx]] += losses[idx]
        loss_i[item_j[idx]] += losses[idx]
        count_u[users[idx]] += 1
        count_i[item_i[idx]] +=1 
        count_i[item_j[idx]] += 1

    for u in range(n_users):
        if count_u[u] > 0:
            loss_u[u] = loss_u[u] / count_u[u]

    for i in range(n_items):
        if count_i[i] > 0:
            loss_i[i] = loss_i[i] / count_i[i]

    cdef np.ndarray[ndim=1, dtype=np.int32_t] sort_u = np.flip(np.argsort(loss_u).astype(np.int32))
    cdef np.ndarray[ndim=1, dtype=np.int32_t] sort_i = np.flip(np.argsort(loss_i).astype(np.int32))

    for idx in range(sort_u.shape[0]):
        rank = idx
        u = sort_u[idx]
        prob_u[u] = 1 / (exp(log(se)/n_samples)**rank)

    for idx in range(sort_i.shape[0]):
        rank = idx
        i = sort_i[idx]
        prob_i[i] = 1 / (exp(log(se)/n_samples)**rank)

    return prob_u / prob_u.sum(), prob_i / prob_i.sum()


def update_history_ex(
    np.ndarray[ndim=1, dtype=np.int32_t] losses,
    np.ndarray[ndim=1, dtype=np.int32_t] users,
    np.ndarray[ndim=1, dtype=np.int32_t] item_i,
    np.ndarray[ndim=1, dtype=np.int32_t] item_j,
    np.ndarray[ndim=1, dtype=np.int32_t] history_u,
    np.ndarray[ndim=1, dtype=np.int32_t] history_i,
    np.ndarray[ndim=1, dtype=np.int32_t] history_count_u,
    np.ndarray[ndim=1, dtype=np.int32_t] history_count_i,
    int step,
    int n_users,
    int n_items,
):
    cdef np.ndarray[np.int32_t] u_count = np.zeros(n_users, dtype=np.int32)
    cdef np.ndarray[np.int32_t] i_count = np.zeros(n_items, dtype=np.int32)
    cdef np.ndarray[np.int32_t] u_out = np.zeros(n_users, dtype=np.int32)
    cdef np.ndarray[np.int32_t] i_out = np.zeros(n_items, dtype=np.int32)
    cdef int i

    with nogil, boundscheck(False):
        for i in range(losses.shape[0]):
            history_count_u[users[i]] += 1
            history_u[users[i]] += losses[i]
            history_count_i[item_i[i]] += 1
            history_count_i[item_j[i]] += 1
            history_i[item_i[i]] += losses[i]
            history_i[item_j[i]] += losses[i]

def parallel_argsort(np.ndarray[ndim=2, dtype=np.float64_t] A, int num_threads=72):
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] A_buff = np.ascontiguousarray(A, dtype = np.float64);
    cdef np.ndarray[np.int32_t,ndim=2] out=np.zeros([A.shape[0],A.shape[1]],dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2, mode='c'] out_buff = np.ascontiguousarray(out, dtype = np.int32);
    cdef double * a_ptr = <double *> A_buff.data;
    cdef int * out_ptr = <int *> out_buff.data;
    cdef int i;
    cdef int k = A.shape[1];
    cdef int thread_id;
    cdef my_index * mm = <my_index *> malloc(num_threads * A.shape[1] * (sizeof(my_index)))
    with nogil, parallel(num_threads=num_threads):
        for i in prange(A.shape[0]):
            thread_id = threadid()
            argsort_row(a_ptr+(i*k) ,  out_ptr + (i*k), mm + (thread_id * k ), k)        
    free(mm)
    return out

@cython.boundscheck(False)
def fast_evaluate(
    np.ndarray[ndim=2, dtype=np.float64_t] P,
    np.ndarray[ndim=2, dtype=np.float64_t] Q,
    np.ndarray[ndim=2, dtype=np.int32_t] train,
    np.ndarray[ndim=2, dtype=np.int32_t] test,
    int topN=10,
    int num_threads=72
):
    cdef int i, j, tpfn, thread_id
    tpfn = P.shape[0] * topN
    cdef np.ndarray[ndim=1, dtype=np.int32_t] max_hits = np.zeros(num_threads,dtype=np.int32)
    cdef np.ndarray[ndim=1, dtype=np.int32_t] tp = np.zeros(num_threads,dtype=np.int32)
    cdef np.ndarray[ndim=2, dtype=np.float64_t] Pred = np.matmul(P, Q.T)
    with nogil, parallel(num_threads=num_threads),boundscheck(False):
        for i in prange(Pred.shape[0]):
            thread_id = threadid()
            for j in prange(Pred.shape[1]):
                if train[i,j] > 0:
                    Pred[i,j] = -1000000.0
                if test[i,j] > 0:
                    max_hits[thread_id]+=1
    cdef np.ndarray[ndim=2,dtype=np.int32_t] rec = parallel_argsort(Pred, num_threads=num_threads)
    with nogil, parallel(num_threads=num_threads):
        thread_id =threadid()
        for i in prange(Pred.shape[0]):
            for j in range(rec.shape[1]-topN, rec.shape[1]):
                if test[i,rec[i,j]] > 0:
                    tp[thread_id]+=1
    return tp.sum() / tpfn, tp.sum() / max_hits.sum()

def update_history(
    np.ndarray[ndim=1, dtype=np.int32_t] losses,
    np.ndarray[ndim=1, dtype=np.int32_t] users,
    np.ndarray[ndim=1, dtype=np.int32_t] item_i,
    np.ndarray[ndim=1, dtype=np.int32_t] item_j,
    np.ndarray[ndim=2, dtype=np.int32_t] history_u,
    np.ndarray[ndim=2, dtype=np.int32_t] history_i,
    np.ndarray[ndim=2, dtype=np.int32_t] history_count_u,
    np.ndarray[ndim=2, dtype=np.int32_t] history_count_i,
    int step,
    int n_users,
    int n_items,
    ):
    cdef np.ndarray[np.int32_t] u_count = np.zeros(n_users, dtype=np.int32)
    cdef np.ndarray[np.int32_t] i_count = np.zeros(n_items, dtype=np.int32)
    cdef np.ndarray[np.int32_t] u_out = np.zeros(n_users, dtype=np.int32)
    cdef np.ndarray[np.int32_t] i_out = np.zeros(n_items, dtype=np.int32)
    cdef int i

    with nogil, boundscheck(False):
        for i in range(losses.shape[0]):
            history_count_u[users[i],step] += 1
            history_u[users[i],step] += losses[i]
            history_count_i[item_i[i],step] += 1
            history_count_i[item_j[i],step] += 1
            history_i[item_i[i], step] += losses[i]
            history_i[item_j[i], step] += losses[i]

def parallel_argsort(np.ndarray[ndim=2, dtype=np.float64_t] A, int num_threads=72):
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] A_buff = np.ascontiguousarray(A, dtype = np.float64);
    cdef np.ndarray[np.int32_t,ndim=2] out=np.zeros([A.shape[0],A.shape[1]],dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2, mode='c'] out_buff = np.ascontiguousarray(out, dtype = np.int32);
    cdef double * a_ptr = <double *> A_buff.data;
    cdef int * out_ptr = <int *> out_buff.data;
    cdef int i;
    cdef int k = A.shape[1];
    cdef int thread_id;
    cdef my_index * mm = <my_index *> malloc(num_threads * A.shape[1] * (sizeof(my_index)))
    with nogil, parallel(num_threads=num_threads):
        for i in prange(A.shape[0]):
            thread_id = threadid()
            argsort_row(a_ptr+(i*k) ,  out_ptr + (i*k), mm + (thread_id * k ), k)        
    free(mm)
    return out

@cython.boundscheck(False)
def fast_evaluate(
    np.ndarray[ndim=2, dtype=np.float64_t] P,
    np.ndarray[ndim=2, dtype=np.float64_t] Q,
    np.ndarray[ndim=2, dtype=np.int32_t] train,
    np.ndarray[ndim=2, dtype=np.int32_t] test,
    int topN=10,
    int num_threads=72
):
    cdef int i, j, tpfn, thread_id
    tpfn = P.shape[0] * topN
    cdef np.ndarray[ndim=1, dtype=np.int32_t] max_hits = np.zeros(num_threads,dtype=np.int32)
    cdef np.ndarray[ndim=1, dtype=np.int32_t] tp = np.zeros(num_threads,dtype=np.int32)
    cdef np.ndarray[ndim=2, dtype=np.float64_t] Pred = np.matmul(P, Q.T)
    with nogil, parallel(num_threads=num_threads),boundscheck(False):
        for i in prange(Pred.shape[0]):
            thread_id = threadid()
            for j in prange(Pred.shape[1]):
                if train[i,j] > 0:
                    Pred[i,j] = -1000000.0
                if test[i,j] > 0:
                    max_hits[thread_id]+=1
    cdef np.ndarray[ndim=2,dtype=np.int32_t] rec = parallel_argsort(Pred, num_threads=num_threads)
    with nogil, parallel(num_threads=num_threads):
        thread_id =threadid()
        for i in prange(Pred.shape[0]):
            for j in range(rec.shape[1]-topN, rec.shape[1]):
                if test[i,rec[i,j]] > 0:
                    tp[thread_id]+=1
    return tp.sum() / tpfn, tp.sum() / max_hits.sum()

@cython.boundscheck(False)
def get_popular_negatives(
    integral[:] user_ids,
    integral[:] item_ids_pos,
    floating[:] item_probs_cumsum,
    int n_items,
    floating[:] rnd_thresholds,
    floating[:, :] train_matrix,
    int n_threads=72,
):
    n_threads = min(n_threads, int(os.cpu_count() * 0.90))
    cdef int j, idx, u, i
    cdef int n_samples = len(user_ids)
    cdef integral[:] item_ids_neg = np.zeros(len(user_ids), dtype=np.int32)

    srand48(time(0))

    with nogil:
        for idx in prange(n_samples, num_threads=n_threads):
            u = user_ids[idx]
            i = item_ids_pos[idx]

            for j in range(n_items):
                if train_matrix[u][j] == 0:
                    if item_probs_cumsum[j] >= rnd_thresholds[idx]:
                        item_ids_neg[idx] = j
                        break

    return np.asarray(item_ids_neg)
