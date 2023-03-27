cimport cython

from libc.math cimport sqrt, log, exp, erfc, M_PI, pow

import numpy as np
cimport numpy as np

np.import_array()


cdef double COEF_ERF[10]
COEF_ERF[0] = 1.00002368
COEF_ERF[1] = 0.37409196
COEF_ERF[2] = 0.09678418
COEF_ERF[3] = -0.18628806
COEF_ERF[4] = 0.27886807
COEF_ERF[5] = -1.13520398
COEF_ERF[6] = 1.48851587
COEF_ERF[7] = -0.82215223
COEF_ERF[8] = 0.17087277
COEF_ERF[9] = -1.26551223  # constant c0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double approx_log_erfc(double x):
    """ Approximates log erfc: 1. - erf when x > 0. """
    cdef:
        double t = 1. / (1. + 0.5 * x)
        double t0 = 1.
        double v = 0.

    for i in range(9):
        t0 *= t
        v += COEF_ERF[i] * t0
    v += COEF_ERF[9]

    return log(t) - x * x + v


@cython.cdivision(True)
cdef double exp_erfc_mul(double x, double z):
    if z <= 0.:
        return exp(x) * erfc(z)
    return exp(x + approx_log_erfc(z))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void emg(double[::1] x, double[::1] param, double[::1] y):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double a = param[0]
        double mu = param[1]
        double s = param[2]
        double lb = param[3]
        double lk = lb / 2.
        double c1 = lb * mu + lb * lb * s * s / 2.
        double c2 = mu + lb * s * s
        double sb = sqrt(2.) * s
        double m, z, g

    for i in range(n):
        m = c1 - lb * x[i]
        z = (c2 - x[i]) / sb
        if z < 0.:
            y[i] = lk * exp(m) * erfc(z) * a
        else:
            g = approx_log_erfc(z) + m
            y[i] = lk * exp(g) * a


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void jacobian_emg(double[::1] x,
                       double a,
                       double u,
                       double s,
                       double b,
                       double[:, ::1] jac):
    """
    Calculates Jacobian matrix for exponentially modified gaussian peak.
    
    Args:
        x: x array
        a: Curve area.
        u: mu
        s: standard deviation sigma
        b: lambda
        jac: Jacobian matrix, for output.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double t1 = sqrt(2. * M_PI)
        double ck = b * b / 2.
        double cj = b * s
        double cu = u + cj * s
        double cq = cj * cj / 2.
        double cz = sqrt(2.) * cj
        double c0 = 0.5
        double c1 = cj / t1
        double c2 = ck
        double c3 = b / (t1 * s)
        double c4 = ck * cj
        double c5_0 = t1 * s * s
        double c5_1 = 4. * ck / t1
        double c6 = b / 2.
        double v, q, m, z, e2, c5

    for i in range(n):
        v = b * (cu - x[i])
        q = v - cq
        z = v / cz
        c5 = v / c5_0 - c5_1
        m = exp_erfc_mul(q, z)
        e2 = exp(q - z * z)
        jac[i, 0] = c6 * m                                  # df/d(A)
        jac[i, 1] = a * (c2 * m - c3 * e2)                  # df/d(u)
        jac[i, 2] = a * (c4 * m + c5 * e2)                  # df/d(sigma)
        jac[i, 3] = a * (c0 * m * (1. + v) - c1 * e2)       # df/d(lambda)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1] estimate_emg_p0(double[::1] x, double[::1] y):
    """ Initializes parameters of EMG. """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double[::1] param = np.zeros(4, dtype=np.float64)
        double m3 = 0.
        double s = 0.
        double nk = 0.
        double m = 0.
        double a = 0.
        double b, g, tau, t

    for i in range(n):
        nk += y[i]
        m += x[i] * y[i]
    m /= nk

    for i in range(1, n):
        a += (y[i - 1] + y[i]) * (x[i] - x[i - 1]) / 2.

    for i in range(n):
        b = x[i] - m
        t = b * b * y[i]
        s += t
        m3 += b * t
    m3 /= nk
    s = sqrt(s / (nk - 1.))
    g = m3 / pow(s, 3.)

    # estimated parameters
    t = pow(g / 2., 1. / 3.)
    tau = s * t
    param[0] = a
    param[1] = m - tau
    param[2] = s
    param[3] = 1. / tau

    return param


cpdef calculate_emg(double[::1] x, double[::1] param):
    """ Calculates EMG. """
    cdef:
        Py_ssize_t n = x.shape[0]
        double[::1] y = np.zeros(n, dtype=np.float64)

    emg(x, param, y)
    return np.asarray(y)


cpdef calculate_jacobian_emg(double[::1] x, double[::1] params):
    cdef:
        Py_ssize_t n = x.shape[0]
        double[:, ::1] jac = np.zeros((n, 4), dtype=np.float64)

    jacobian_emg(x, params[0], params[1], params[2], params[3], jac)

    return np.asarray(jac)
