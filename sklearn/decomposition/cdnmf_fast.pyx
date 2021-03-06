# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Mathieu Blondel, Tom Dupre la Tour
# License: BSD 3 clause

cimport cython
from libc.math cimport fabs


def _update_cdnmf_fast(double[:, ::1] W, double[:, ::1] Ht,
                       double[:, :] HHt, double[:, :] WtW,
                       double[:, :] XHt, double[:, :] XtW, double[:, :] X,
                       Py_ssize_t[::1] permutation, int[::1] w_free_cols):
    cdef double violation = 0
    cdef Py_ssize_t n_components = W.shape[1]
    cdef Py_ssize_t n_samples = W.shape[0]  # n_features for H update
    cdef Py_ssize_t n_features = Ht.shape[0] 
    cdef double grad, pg, hess
    cdef Py_ssize_t i, r, s, t
    cdef double old_val, iter_change

    with nogil:
        for s in range(n_components):
            t = permutation[s]

            if w_free_cols[t]==1:
                for i in range(n_samples):
                    # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                    grad = -XHt[i, t]

                    for r in range(n_components):
                        grad += HHt[t, r] * W[i, r]

                    # projected gradient
                    pg = min(0., grad) if W[i, t] == 0 else grad
                    violation += fabs(pg)

                    # Hessian
                    hess = HHt[t, t]

                    if hess != 0:
                        old_val = W[i, t]
                        W[i, t] = max(W[i, t] - grad / hess, 0.)
                        iter_change = W[i, t] - old_val

                        # update WtW and XtW
                        for r in range(n_components):
                            if r == t:
                                WtW[r, t] += W[i, t]**2 - old_val**2
                            else:
                                WtW[r, t] += W[i, r] * iter_change
                        
                        for r in range(n_features):
                            XtW[r, t] += X[i, r] * iter_change  
            else:
                for i in range(n_features):
                    grad = -XtW[i, t] ###

                    for r in range(n_components):
                        grad += WtW[t, r] * Ht[i, r] ###

                    pg = min(0., grad) if Ht[i, t] == 0 else grad 
                    violation += fabs(pg)
    
                    hess = WtW[t, t] ###

                    if hess != 0:
                        old_val = Ht[i, t]
                        Ht[i, t] = max(Ht[i, t] - grad / hess, 0.)
                        iter_change = Ht[i, t] - old_val
                        
                        # update HHt and XHt 
                        for r in range(n_components):
                            if r == t:
                                HHt[r, t] += Ht[i, t]**2 - old_val**2
                            else:
                                HHt[r, t] += Ht[i, r] * iter_change
    
                        for r in range(n_samples):
                            XHt[r, t] += X[r, i] * iter_change
    return violation
