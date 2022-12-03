import numpy as np
cimport numpy as np
from cython import boundscheck
from cython.parallel import prange, parallel, threadid
from libc.math cimport exp, log
from libc.stdlib cimport malloc, free
# from libcpp cimport bool
from libc.stdio cimport printf

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



def bpr_update(
    np.ndarray[np.float64_t, ndim=2] U not None,
    np.ndarray[np.float64_t, ndim=2] I not None,
    np.ndarray[np.int32_t, ndim=1] users not None,
    np.ndarray[np.int32_t, ndim=1] pos_items not None,
    np.ndarray[np.int32_t, ndim=1] neg_items not None,
    np.int32_t n_samples,
    np.float32_t lr,
    np.float32_t reg,
    bint update_user_factors,
    bint update_item_factors,
    bint non_negative
):
    # lr=0.01
    # reg=0.01
    cdef Py_ssize_t i, j, u, idx, jdx, n_factors
    cdef double u_factor, i_factor, j_factor
    cdef float error, prediction_ui, prediction_uj

    n_factors = I.shape[1]
    for idx in range(0, n_samples):

        u = users[idx]
        i = pos_items[idx]
        j = neg_items[idx]

        prediction_ui = 0
        prediction_uj = 0
        for jdx in range(n_factors):
            prediction_ui += U[u, jdx] * I[i, jdx]
            prediction_uj += U[u, jdx] * I[j, jdx]

        error = prediction_ui - prediction_uj
        error = 1 / (1 + exp(-1 * error))

        for jdx in range(n_factors):
            u_factor = U[u, jdx] + 0
            i_factor = I[i, jdx] + 0
            j_factor = I[j, jdx] + 0

            if update_user_factors:
                U[u, jdx] += lr * (error * (i_factor - j_factor) - (reg * u_factor))
                if non_negative:
                    U[u, jdx] = max(U[u, jdx], 0)
            if update_item_factors:
                I[i, jdx] += lr * (error * u_factor - (reg * i_factor))
                I[j, jdx] += lr * ((error * (- u_factor)) - (reg * j_factor))
                if non_negative:
                    I[i, jdx] = max(I[i, jdx], 0)
                    I[j, jdx] = max(I[j, jdx], 0)

    return U, I

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


def bpr_update_return_loss(np.ndarray[np.float64_t, ndim=2] I not None,
                        np.ndarray[np.float64_t, ndim=2] U not None,
                        np.ndarray[np.int32_t, ndim=1] users not None,
                        np.ndarray[np.int32_t, ndim=1] pos_items not None,
                        np.ndarray[np.int32_t, ndim=1] neg_items not None,
                        np.int32_t n_samples,
                        np.float32_t lr,
                        np.float32_t reg,
                        np.int32_t update_users,
                        ):

    lr=0.01
    reg=0.01
    cdef Py_ssize_t user_id, i,j,u,idx, jdx, n_factors, node_id
    cdef double grad_i, grad_j, grad_u, u_factor, i_factor, j_factor, sigmoid
    cdef float diff,loss
    loss = 0
    
    n_factors = I.shape[1]
    
    with nogil, boundscheck(False):
        for idx in range(0,n_samples):

            u = users[idx]
            i = pos_items[idx]
            j = neg_items[idx]
            
            diff=0
            for jdx in range(n_factors):
                diff += (U[u,jdx] * I[i, jdx]) - (U[ u,jdx] * I[j, jdx])
            
            loss+=-log(1 / (1 + exp(-diff)))
            diff = 1 / (1 + exp(diff))
            
            #printf("\n%lf",diffs[node_id])
            for jdx in range(n_factors):

                u_factor = U[ u, jdx] + 0
                i_factor = I[i, jdx] + 0
                j_factor = I[ j, jdx] + 0
                if update_users > 0:
                    U[u, jdx] += lr * (diff * (i_factor - j_factor) - (reg * u_factor))

                I[ i ,jdx] += lr * (diff *  u_factor - (reg * i_factor ))

                I[j, jdx] += lr * ( (diff * (- u_factor)) - (reg * j_factor ))

            diff=0
    return U, I, loss/n_samples




def negative_probs_for_similar_items(
    np.ndarray[ndim=2, dtype=np.float64_t] item_similarity,
    np.ndarray[ndim=2, dtype=np.float32_t] train_mat,
    np.ndarray[ndim=1,dtype=np.int32_t] user_ids,
    np.ndarray[ndim=1,dtype=np.int32_t] item_i,
    np.ndarray[ndim=1,dtype=np.float64_t] random_seeds,
    double exponent,
    int num_threads=72,
    
    ):

    cdef np.ndarray[ndim=1, dtype=np.int32_t] item_j = np.zeros(len(item_i),dtype=np.int32)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] totals =np.zeros(len(item_i),dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] ms =np.zeros(len(item_i),dtype=np.float64)
    cdef int i, u, j,idx
    cdef float total,r,m
    srand48(time(0))
    with nogil, boundscheck(False):
        for idx in prange(item_i.shape[0], num_threads=num_threads):
            
        #for idx in range(item_i.shape[0]):
            u = user_ids[idx]
            i = item_i[idx]
            
            for j in range(item_similarity.shape[1]):

                if train_mat[u, j] ==0.0:
                    totals[idx] += item_similarity[i,j] ** exponent

            r = random_seeds[idx]
            ms[idx]=0
            if totals[idx] == 0.0:
                item_j[idx]=-1
                continue
            for j in range(item_similarity.shape[1]):

                if train_mat[u,j]==0:
                    ms[idx] += (item_similarity[i,j] ** exponent) / totals[idx]
                    if ms[idx] >= r:
                        item_j[idx] = j
                        break
    return item_j
                
            


def compute_probabilities(
    np.ndarray[ndim=2, dtype=np.int32_t] u_history,
    np.ndarray[ndim=2, dtype=np.int32_t] i_history,
    np.ndarray[ndim=2, dtype=np.int32_t] u_count,
    np.ndarray[ndim=2, dtype=np.int32_t] i_count,
    double selection_pressure,
    int q,
    int step):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] u_out = np.zeros(u_history.shape[0], dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] i_out = np.zeros(i_history.shape[0], dtype=np.float64)
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
            u_out[user_idx] = 1/((exp(log(selection_pressure/i_history.shape[0])))**u_out[user_idx])

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
    cdef Py_ssize_t user_id, i, j, u, idx, jdx, n_factors, node_id
    cdef Py_ssize_t batch_idx, prev_batch_idx, kdx
    cdef double cur_batch_size, cur_loss
    cdef double denominator
    cdef double diff
    cdef double grad_i, grad_j, grad_u
    cdef double i_factor, j_factor
    cdef double loss
    cdef double neg_predict
    cdef double noise_i, noise_j, noise_u
    cdef double sigmoid
    cdef double u_factor 
    cdef np.ndarray[ndim=1,dtype=np.float64_t] U_updates = np.zeros(len(U), dtype=np.float64)
    cdef np.ndarray[ndim=1,dtype=np.float64_t] I_updates = np.zeros(len(I), dtype=np.float64)
    cdef np.ndarray[ndim=1,dtype=np.int32_t] losses = np.zeros(n_samples, dtype=np.int32)

    n_factors = I.shape[1]

    with boundscheck(False):
        loss = 0
        batch_idx = batch_size
        prev_batch_idx = 0
        for idx in range(0, n_samples):
            diff = 0
            u = users[idx]
            i = pos_items[idx]
            j = neg_items[ idx]

            U_updates[u] += 1
            I_updates[i] += 1
            I_updates[j] += 1

            for jdx in range(n_factors):
                diff += (U[u, jdx] * I[i, jdx]) - (U[u,jdx] * I[j, jdx])

            loss += -log(1 / (1 + exp(-diff)))
            
            if diff > diff_limit:
                losses[idx] = 1
            else:
                losses[idx] = 0

            diff = 1 / (1 + exp(diff))

            for jdx in range(n_factors):
                u_factor = U[u, jdx] + 0
                i_factor = I[i, jdx] + 0
                j_factor = I[j, jdx] + 0
                U_scratch[u, jdx] += lr_user * (diff * (i_factor - j_factor) - (reg * u_factor))
                I_scratch[i, jdx] += lr_item * (diff *  u_factor - (reg * i_factor ))
                I_scratch[j, jdx] += lr_item *  ( (diff * (- u_factor)) - (reg * j_factor ))
                
            if idx == n_samples - 1 or idx == batch_idx:
                #printf("Here....")
                #printf("\n %d %d %d %d", prev_batch_idx, batch_idx, idx, n_samples[node_id])
                cur_batch_size = min(batch_size, n_samples-prev_batch_idx)
                for u in range(U_updates.shape[0]):
                    if U_updates[ u] > 0:
                        for jdx in range(n_factors):
                            # U[ u, jdx] += U_scratch[ u, jdx] #/ cur_batch_size # U_updates[node_id,u]
                            U[ u, jdx] += U_scratch[ u, jdx] / cur_batch_size # U_updates[node_id,u]
                            # U[ u, jdx] += U_scratch[ u, jdx] / U_updates[node_id,u]
                            U_scratch[ u, jdx] = 0
                        U_updates[ u] = 0
                for i in range(I_updates.shape[0]):
                    if I_updates[ i] > 0:
                        for jdx in range(n_factors):
                            #printf("\n%lf",  I_scratch[node_id, i, jdx] / cur_batch_size)
                            #I[ i, jdx] += I_scratch[ i, jdx] #/ cur_batch_size #I_updates[node_id,i]
                            I[ i, jdx] += I_scratch[ i, jdx] / cur_batch_size #I_updates[node_id,i]
                            # I[ i, jdx] += I_scratch[ i, jdx] / I_updates[node_id,i]
                            I_scratch[ i, jdx] = 0
                        I_updates[ i] = 0
                prev_batch_idx = batch_idx
                batch_idx = prev_batch_idx + batch_size
        loss = loss / n_samples
    return U, I, loss, losses
