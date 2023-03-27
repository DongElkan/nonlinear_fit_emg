cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

import numpy as np
cimport numpy as np

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double norm(double[::1] x):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double s = 0.

    for i in range(n):
        s += x[i] * x[i]

    return sqrt(s)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void col_norm(double[:, ::1] x, double[::1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t i, j
        double s

    for j in range(p):
        s = 0.
        for i in range(n):
            s += x[i, j] * x[i, j]
        c[j] = s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void row_norm(double[:, ::1] x, double[::1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i

    for i in range(n):
        c[i] = norm(x[i])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int qr(double[:, ::1] a, int[::1] piv_index, double[::1] beta):
    """
    Decomposes matrix A to orthogonal matrix Q and upper triangular
    matrix R using Householder transformation with column pivoting. The
    upper triangular part of A is overwritten.
    """
    cdef:
        Py_ssize_t n = a.shape[0]
        Py_ssize_t p = a.shape[1]
        Py_ssize_t i, j
        int rnk = -1
        int k
        double * col_norms = <double *> malloc(p * sizeof(double))
        double * house_vec = <double *> malloc(n * sizeof(double))
        double * va = <double *> malloc(p * sizeof(double))
        double m_cn = 0.
        double s, t, g, b, u, v1

    for j in range(p):
        s = 0.
        for i in range(n):
            s += a[i, j] * a[i, j]
        col_norms[j] = s
        if s > m_cn:
            m_cn = s
            k = <int> j

    while m_cn > 0. and rnk < p - 1:
        rnk += 1

        piv_index[rnk] = k
        # column pivoting
        if k != rnk:
            for i in range(n):
                t = a[i, rnk]
                a[i, rnk] = a[i, k]
                a[i, k] = t
            t = col_norms[k]
            col_norms[k] = col_norms[rnk]
            col_norms[rnk] = t

        # Householder vector: Algorithm 5.1.1 in Matrix Computations (4th Ed)
        g = 0.
        for i in range(rnk + 1, n):
            g += a[i, rnk] * a[i, rnk]
        t = a[rnk, rnk]
        house_vec[rnk] = 1.
        if g == 0.:
            if t >= 0.:
                b = 0.
            else:
                b = -2.
            for i in range(rnk + 1, n):
                house_vec[i] = 0.
        else:
            u = sqrt(g + t * t)
            if t <= 0.:
                v1 = t - u
            else:
                v1 = -g / (t + u)
            b = 2. * v1 * v1 / (g + v1 * v1)
            for i in range(rnk + 1, n):
                house_vec[i] = a[i, rnk] / v1
        beta[rnk] = b

        # Householder transformation
        for i in range(rnk, p):
            s = 0.
            for j in range(rnk, n):
                s += house_vec[j] * a[j, i]
            va[i] = s
        if rnk < p - 1:
            for i in range(rnk, n):
                t = b * house_vec[i]
                for j in range(rnk + 1, p):
                    a[i, j] = a[i, j] - t * va[j]

        a[rnk, rnk] -= b * house_vec[rnk] * va[rnk]
        for i in range(rnk + 1, n):
            a[i, rnk] = house_vec[i]

        # update norms
        for i in range(rnk + 1, p):
            col_norms[i] -= a[rnk, i] * a[rnk, i]
        m_cn = col_norms[i]
        k = <int> i
        for i in range(rnk + 1, p):
            if col_norms[i] > m_cn:
                m_cn = col_norms[i]
                k = <int> i

    free(col_norms)
    free(house_vec)
    free(va)

    return rnk + 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void solve_linear(double[:, ::1] a, double[::1] z, double[::1] x):
    """
    Solves linear equation system by QR decomposition. For linear
    system Ax = z, and QR decomposition of A:
        A = QR
    Solve the equations sequentially:
        u = Q'z
        Rx = u

    Args:
        a: Matrix A.
        z: Dependent variable.
        x: Solution

    """
    cdef:
        Py_ssize_t n = a.shape[0]
        Py_ssize_t p = a.shape[1]
        Py_ssize_t i, j, k
        int[::1] piv = np.zeros(p, dtype=np.int32)
        int rnk
        double[::1] beta = np.zeros(p, dtype=np.float64)
        double s, b, vz, t

    rnk = qr(a, piv, beta)
    if rnk != p:
        raise ValueError("The coefficient matrix A is singular.")

    # solve the linear equations using Algorithm 5.3.2, Matrix Computations
    # (4th ed), with column pivoting.
    for j in range(p):
        vz = z[j]
        for i in range(j + 1, n):
            vz += a[i, j] * z[i]
        b = beta[j] * vz
        z[j] -= b
        for i in range(j + 1, n):
            z[i] -= b * a[i, j]

    # solve Rx = z
    for i in range(1, p + 1):
        j = p - i
        s = 0.
        for k in range(j + 1, p):
            s += a[j, k] * x[k]
        x[j] = (z[j] - s) / a[j, j]

    # pivot to original order
    for i in range(1, p + 1):
        j = p - i
        if piv[j] != j:
            k = piv[j]
            t = x[k]
            x[k] = x[j]
            x[j] = t


cpdef test_qr(double[:, ::1] a):
    cdef:
        Py_ssize_t p = a.shape[1]
        int[::1] piv = np.zeros(p, dtype=np.int32)
        double[::1] beta = np.zeros(p, dtype=np.float64)
    qr(a, piv, beta)
    return np.asarray(a), np.asarray(piv), np.asarray(beta)


cpdef test_solve_linear(double[:, ::1] a, double[::1] z):
    cdef:
        Py_ssize_t n = a.shape[0]
        Py_ssize_t p = a.shape[1]
        double[::1] x = np.zeros(p, dtype=np.float64)

    solve_linear(a, z, x)
    return np.asarray(x)
