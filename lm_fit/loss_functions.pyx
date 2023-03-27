cimport cython

from .emg cimport jacobian_emg, emg


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void eval_func(double[::1] x, double[::1] y, double[::1] param, double[::1] err, int k):
    """
    Evaluates function according to index k. This is used for
    customizing function.
    Index k:
        1. To fit EMG (exponentially modified gaussian) function.

    Args:
        x: Independent variable x
        y: Dependent variable y
        param: Parameter for the function
        err: For output, evaluation at current x and param.
        k: Index

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i

    if k == 1:
        emg(x, param, err)

    for i in range(n):
        err[i] = y[i] - err[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void eval_jac(double[::1] x, double[::1] param, double[:, ::1] jac, int k):
    """
    Evaluates Jacobian matrix according to index k, based on the
    function defined in "eval_func".

    Args:
        x: x
        param: Parameter for the function
        jac: For output, Jacobian matrix.
        k: Index

    """
    if k == 1:
        jacobian_emg(x, param[0], param[1], param[2], param[3], jac)
