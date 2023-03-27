"""
This module performs nonlinear least squares using Levenberg-Marquardt
algorithm.

References:
    J. J. Moré. The Levenberg-Marquardt algorithm: Implementation
    and theory. Numerical Analysis. 1978, 105-116.

"""
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrt

import numpy as np
cimport numpy as np

from .linalg cimport qr, norm
from .loss_functions cimport eval_func, eval_jac

np.import_array()

DTYPE = np.float64


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void pivot_x(double[::1] x, int[::1] pivot_index):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double t

    for i in range(n):
        if pivot_index[i] != i:
            t = x[i]
            x[i] = x[pivot_index[i]]
            x[pivot_index[i]] = t


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void back_pivot_x(double[::1] x, int[::1] pivot_index):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i, j
        double t

    for i in range(1, n + 1):
        j = n - i
        if pivot_index[j] != j:
            t = x[j]
            x[j] = x[pivot_index[j]]
            x[pivot_index[j]] = t


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void qrsolve(double[:, ::1] r, double[::1] diag, double[::1] qtb, double[::1] x, double[::1] sdiag):
    """
    Given an m by n matrix a, an n by n diagonal matrix d, and an
    m-vector b, the problem is to determine an x which solves the
    system
           a*x = b ,     d*x = 0 ,
    in the least squares sense.
    
    This function completes the solution of the problem if it is
    provided with the necessary information from the qr factorization,
    with column pivoting, of a. that is, if a*p = q*r, where p is a
    permutation matrix, q has orthogonal columns, and r is an upper
    triangular matrix with diagonal elements of non-increasing
    magnitude, then qrsolv expects the full upper triangle of r, the
    permutation matrix p, and the first n components of q'*b. the
    system a*x = b, d*x = 0, is then equivalent to
            r*z = q'*b ,  p'*d*p*z = 0 ,
    where x = p*z. if this system does not have full rank, then a least
    squares solution is obtained. on output qrsolv also provides an
    upper triangular matrix s such that
            p'*(a'*a + d*d)*p = s'*s .
    s is computed within qrsolv.
    
    Args:
        r: Upper triangular matrix from QR factorization.
        diag: Diagonal elements of R.
        qtb: First n elements of Q'*b
        x: An output array of length n which contains the least squares
            solution of the system a*x = b, d*x = 0.
        sdiag: An output array of length n which contains the diagonal
            elements of the upper triangular matrix s
    
    """
    cdef:
        Py_ssize_t p = r.shape[0]
        Py_ssize_t i, j, k
        int rnk = <int> p
        double * qtb_c = <double *> malloc(p * sizeof(double))
        double qtbj, cos, sin, cotan, t, s

    for j in range(p):
        for i in range(j, p):
            r[i, j] = r[j, i]
        x[j] = r[j, j]
        qtb_c[j] = qtb[j]

    # eliminate the diagonal matrix d using a givens rotation
    for j in range(p):
        # perpare the row of d to be eliminated, locating the diagonal
        # element using p from the QR factorization
        if diag[j] != 0:
            for k in range(j, p):
                sdiag[k] = 0.
            sdiag[j] = diag[j]

            # the transformation to eliminate the row of d modify only
            # a single element of q'b beyond the first n, which is
            # initially zero.
            qtbj = 0.
            for k in range(j, p):
                # determine a givens rotation which eliminates the
                # appropriate element in the current row of d.
                if sdiag[k] != 0.:
                    cotan = r[k, k] / sdiag[k]
                    if fabs(cotan) >= 1.:
                        cos = 1. / (sqrt(0.25 + 0.25 / (cotan * cotan)) * 2.)
                        sin = cos / cotan
                    else:
                        sin = 1. / (sqrt(0.25 + 0.25 * cotan * cotan) * 2.)
                        cos = sin * cotan

                    # compute the modified diagonal element of r and the
                    # modified element of (q'b, 0)
                    r[k, k] = cos * r[k, k] + sin * sdiag[k]
                    t = cos * qtb_c[k] + sin * qtbj
                    qtbj = cos * qtbj - sin * qtb_c[k]
                    qtb_c[k] = t

                    # accumulate the transformation in the row of s
                    for i in range(k + 1, p):
                        t = cos * r[i, k] + sin * sdiag[i]
                        sdiag[i] = cos * sdiag[i] - sin * r[i, k]
                        r[i, k] = t

        # store the diagonal element of s and restore the corresponding
        # diagonal element of r
        sdiag[j] = r[j, j]
        r[j, j] = x[j]

    # solve the triangular system for z, if the system is singular, then
    # obtain the least squares solution.
    for j in range(p):
        if sdiag[j] == 0. and rnk == p:
            rnk = j - 1
        if rnk < p:
            qtb_c[j] = 0.

    if rnk >= 0:
        for k in range(1, rnk + 1):
            j = rnk - k
            s = 0.
            for i in range(j + 1, rnk):
                s += r[i, j] * qtb_c[i]
            qtb_c[j] = (qtb_c[j] - s) / sdiag[j]

    for j in range(p):
        x[j] = qtb_c[j]

    free(qtb_c)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double lmpar(double[:, ::1] r, double[::1] diag, double[::1] qtb, double delta, double[::1] x, double[::1] sdiag):
    """
    Given an m by n matrix a, an n by n nonsingular diagonal matrix d,
    an m-vector b, and a positive number delta, the problem is to
    determine a value for the parameter par such that if x solves the
    system
            a*x = b,     sqrt(par)*d*x = 0
    in the least squares sense, and dxn is the euclidean norm of d*x,
    then either par is zero and
            (dxnorm-delta) <= 0.1*delta,
    or par is positive and
            abs(dxnorm-delta) <= 0.1*delta.
    This function completes the solution of the problem if it is
    provided with the necessary information from the QR factorization,
    with column pivoting, of a. that is, if a*p = q*r, where p is a
    permutation matrix, q has orthogonal columns, and r is an upper
    triangular matrix with diagonal elements of nonincreasing
    magnitude, then lmpar expects the full upper triangle of r, the
    permutation matrix p, and the first n components of q'*b. on output
    lmpar also provides an upper triangular matrix s such that
            p'*(a*a + par*d*d)*p = s *s.
    Where s is employed within lmpar.

    Only a few iterations are generally needed for convergence of the
    algorithm. if, however, the limit of 10 iterations is reached, then
    the output par will contain the best value obtained so far.

    Args:
        r: an p by p array. on input the full upper triangle must
            contain the full upper triangle of the matrix R. On output
            the full upper triangle is unaltered, and the strict lower
            triangle contains the strict upper triangle (transposed) of
            the upper triangle matrix s.
        diag: an input array of length p which must contain the
            diagonal elements of the matrix d.
        qtb: an input array of length n which must contain the first p
            elements of the vector Q'b.
        delta: a positive input variable which specifies an upper bound
            on the euclidean norm of d*x.
        x: an output array of length p which contains the least squares
            solution of the system a*x = b, sqrt(par)*d*x = 0, for the
            output par.
        sdiag: Diagonal elements of the upper triangular matrix s.

    """

    cdef:
        Py_ssize_t p = r.shape[1]
        Py_ssize_t i, j, k
        int rnk = <int> p
        int it = 0
        double * qtb_c = <double *> malloc(p * sizeof(double))
        double * wa1 = <double *> malloc(p * sizeof(double))
        double * dx = <double *> malloc(p * sizeof(double))
        double dwarf = 1e-15
        double par = 0.
        double parl = 0.
        double t, s, a, dxn, fp, gn
        double paru

    # compute and store in x the gauss-newton direction. if the
    # jacobian is rank-deficient, obtain the least squares solution.
    for j in range(p):
        qtb_c[j] = qtb[j]
        if r[j, j] == 0. and rnk == p:
            rnk = j - 1
        if rnk < p:
            qtb_c[j] = 0.

    if rnk >= 0:
        if rnk == p:
            rnk -= 1
        for k in range(rnk + 1):
            j = rnk - k
            qtb_c[j] /= r[j, j]
            t = qtb_c[j]
            for i in range(j):
                qtb_c[i] -= r[i, j] * t

    for j in range(p):
        x[j] = qtb_c[j]

    # initialize the iteration counter. evaluate the function at the
    # origin, and test for acceptance of the gauss-newton direction.
    s = 0.
    for j in range(p):
        t = diag[j] * x[j]
        dx[j] = t
        s += t * t
    dxn = sqrt(s)
    fp = dxn - delta
    if fp <= 0.1 * delta:
        return par

    # if the jacobian is not rank deficient, the newton step provides a
    # lower bound, parl, for the zero of the function. otherwise set
    # this bound to zero.
    if rnk == p - 1:
        for j in range(p):
            wa1[j] = diag[j] * (dx[j] / dxn)
        a = 0.
        for j in range(p):
            s = 0.
            for i in range(j):
                s += r[i, j] * wa1[i]
            t = (wa1[j] - s) / r[j, j]
            wa1[j] = t
            a += t * t
        parl = (fp / delta) / a

    # calculate an upper bound, paru, for the zero of the function.
    a = 0.
    for j in range(p):
        s = 0.
        for i in range(j + 1):
            s += r[i, j] * qtb[i]
        wa1[j] = s / diag[j]
        a += wa1[j] * wa1[j]
    gn = sqrt(a)
    paru = gn / delta
    if paru == 0.:
        paru = dwarf / min(delta, 0.1)

    # if the input par lies outside of the interval (parl, paru),
    # set par to the closer endpoint.
    if parl > par:
        par = parl
    if par > paru:
        par = paru
    if par == 0.:
        par = gn / dxn

    # beginning of an iteration.
    while it < 10:
        it = it + 1
        # evaluate the function at the current value of par.
        if par == 0.:
            par = max(dwarf, 0.001 * paru)
        t = sqrt(par)
        for j in range(p):
            wa1[j] = t * diag[j]
        qrsolve(r, diag, qtb, x, sdiag)
        s = 0.
        for j in range(p):
            dx[j] = diag[j] * x[j]
            s += dx[j] * dx[j]
        dxn = sqrt(s)
        t = fp
        fp = dxn - delta

        # if the function is small enough, accept the current value
        # of par. also test for the exceptional cases where parl
        # is zero or the number of iterations has reached 10.
        if (fabs(fp) <= 0.1 * delta or parl == 0.) and fp <= t < 0.:
            break

        # compute the newton correction.
        for j in range(p):
            wa1[j] = diag[j] * (dx[j] / dxn)
        a = 0.
        for j in range(p):
            wa1[j] /= sdiag[j]
            t = wa1[j]
            for i in range(j + 1, p):
                wa1[i] -= r[i, j] * t
            a += t * t
        parc = (fp / delta) / a

        # depending on the sign of the function, update parl or paru.
        if fp > 0.:
            parl = max(parl, par)
        elif fp < 0.:
            paru = min(paru, par)

        # compute an improved estimate for par.
        par = max(parl, par + parc)

    free(qtb_c)
    free(wa1)
    free(dx)

    return par


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void lma(double[::1] x, double[::1] y, double[::1] param, int fun_index):
    """
    Optimizes parameters using Levenberge-Marquardt algorithm.
    
    Args:
        x: Independent variable.
        y: Dependent variable.
        param: Initial parameters, will be rewritten and output.
        fun_index: Index of function to minimize. The function is
            customizable which is indicated by the index. The Jacobian
            matrix is also evaluated by calling the function "eval_jac"
            using the same index.
            1: To fit EMG (exponentially modified gaussian) function.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = param.shape[0]
        Py_ssize_t i, j, ji, k
        int[::1] piv = np.zeros(p, dtype=np.int32)
        int max_iter = 100
        int it = 1
        int rnk
        bint terminate = 0
        double * tmp_val = <double *> malloc(n * sizeof(double))
        double[:, ::1] jac = np.zeros((n, p), dtype=DTYPE)
        double[::1] fval = np.zeros(n, dtype=DTYPE)
        double[::1] beta = np.zeros(p, dtype=DTYPE)
        double[::1] dx = np.zeros(p, dtype=DTYPE)
        double[::1] param_n = np.zeros(p, dtype=DTYPE)
        double[::1] rdiag = np.zeros(p, dtype=DTYPE)
        double[::1] qtf = np.zeros(p, dtype=DTYPE)
        double[::1] sdiag = np.zeros(p, dtype=DTYPE)
        double[::1] jac_col_norm = np.zeros(p, dtype=DTYPE)
        double rtol = 0.0001
        double gtol = 0.0001
        double ftol = 0.000001
        double ptol = 0.0001
        double eps = 1e-15
        double factor = 100.
        double ratio = 0.
        double lb, fn, fn_n, s, t, pn, gn, sj, t1, t2, xn
        double actred, prered, dirder, par

    eval_func(x, y, param, fval, fun_index)
    fn = norm(fval)

    # initialize levenberg-marquardt parameter and iteration counter.
    par = 0.
    # beginning of the outer loop.
    while it < max_iter:
        print(it)
        print(np.asarray(param))
        eval_func(x, y, param, fval, fun_index)
        eval_jac(x, param, jac, fun_index)
        for j in range(p):
            s = 0.
            for i in range(n):
                s += jac[i, j] * jac[i, j]
            jac_col_norm[j] = sqrt(s)

        # compute the qr factorization of the jacobian.
        rnk = qr(jac, piv, beta)

        # make the order of parameters consistent with Jacobian
        # columns after QR factorization and column pivoting
        pivot_x(param, piv)
        pivot_x(jac_col_norm, piv)

        # on the first iteration, calculate the norm of the scaled x and
        # initialize the step bound delta, and scale according to the norms
        # of the columns of the initial jacobian
        if it == 1:
            xn = 0.
            for j in range(p):
                t = jac_col_norm[j]
                if t == 0.:
                    t = 1.
                rdiag[j] = t
                t *= param[j]
                xn += t * t
            xn = sqrt(xn)
            delta = factor * xn
            if delta == 0.:
                delta = factor

        # form q' * fval and store the first n components in qtf.
        for i in range(n):
            tmp_val[i] = fval[i]
        for j in range(p):
            s = fval[j]
            for i in range(j + 1, n):
                s += jac[i, j] * tmp_val[i]
            t = beta[j] * s
            tmp_val[j] -= t
            for i in range(j + 1, n):
                tmp_val[i] -= jac[i, j] * t
            qtf[j] = tmp_val[j]

        # compute the norm of the scaled gradient.
        gn = 0.
        if fn != 0.:
            for j in range(p):
                if jac_col_norm[j] != 0.:
                    s = 0.
                    for i in range(j + 1):
                        s = s + jac[i, j] * (qtf[i] / fn)
                    t = fabs(s / jac_col_norm[j])
                    if t > gn:
                        gn = t
        print("gnorm: ", gn)

        # test for convergence of the gradient norm.
        if gn <= gtol:
            back_pivot_x(param, piv)
            break

        # rescale if necessary
        for j in range(p):
            rdiag[j] = max(rdiag[j], jac_col_norm[j])

        # beginning of the inner loop.
        while ratio <= rtol:
            # determine the levenberg-marquardt parameter.
            # :: sdiag is only used in lmpar and subfunction qrsolv
            print("current start")
            par = lmpar(jac, rdiag, qtf, delta, dx, sdiag)
            print("LM parameter: ", par)

            # store the direction p and x + p. calculate the norm of p.
            s = 0.
            for j in range(p):
                dx[j] = -dx[j]
                param_n[j] = param[j] + dx[j]
                t = rdiag[j] * dx[j]
                s += t * t
            pn = sqrt(s)
            print("pnorm: ", pn)

            # on the first iteration, adjust the initial step bound.
            if it == 1:
                delta = min(delta, pn)

            # evaluate the function at x + p and calculate its norm.
            eval_func(x, y, param_n, fval, fun_index)
            fn_n = norm(fval)
            print("f norms: ", fn_n, fn)

            # compute the scaled actual reduction.
            actred = -1.
            t = fn_n / fn
            if t < 10.:
                actred = 1. - t * t

            # compute the scaled predicted reduction and the scaled
            # directional derivative.
            s = 0.
            for j in range(p):
                sj = 0.
                t = dx[j]
                for i in range(j + 1):
                    sj += jac[i, j] * t
                s += sj * sj
            t1 = s / (fn * fn)
            t2 = sqrt(par) * pn / fn
            prered = t1 + t2 * t2 * 2.
            dirder = -(t1 + t2 * t2)
            print("reds: ", prered, dirder, actred)

            # compute the ratio of the actual to the predicted reduction.
            ratio = 0.
            if prered != 0.:
                ratio = actred / prered
            print("ratio: ", ratio)

            # update the step bound.
            if ratio <= 0.25:
                if actred >= 0.:
                    t = 0.5
                else:
                    t = 0.5 * dirder / (dirder + 0.5 * actred)
                if 0.1 * fn_n >= fn or t < 0.1:
                    t = 0.1
                delta = t * min(delta, pn * 10.)
                par /= t
            elif par == 0. or ratio >= 0.75:
                delta = pn * 2.
                par *= 0.5
            print("delta:", par, t, delta)

            # test for successful iteration.
            if ratio >= rtol:
                # successful iteration. update x, fvec, and their norms.
                s = 0.
                for j in range(p):
                    param[j] = param_n[j]
                    t = rdiag[j] * param[j]
                    s += t * t
                xn = sqrt(s)
                fn = fn_n
                it = it + 1

            print("current end")

            # tests for convergence.
            if ((fabs(actred) <= ftol and prered <= ftol and ratio / 2. <= 1.)
                    or delta <= ptol * xn or gn <= eps):
                terminate = 1
                break

        back_pivot_x(param, piv)

        if terminate:
            break

    free(tmp_val)
    print("exit")


cpdef void test_qrsolve(double[:, ::1] r,
                        double[::1] diag,
                        double[::1] qtb,
                        double[::1] x,
                        double[::1] sdiag):
    qrsolve(r, diag, qtb, x, sdiag)
