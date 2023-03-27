from .lm cimport lma

import numpy as np


cpdef nl_fit(double[::1] x, double[::1] y, double[::1] param, int index):
    """
    Non-linear fit using Levenberg-Marquardt algorithm.
    
    Args:
        x: x
        y: y
        param: Initial parameters. They will be updated during the
            optimization. 
        index: Index for function.
            1. EMG function.

    """
    lma(x, y, param, index)
    print("exit LM, parameters are: ", np.asarray(param))
