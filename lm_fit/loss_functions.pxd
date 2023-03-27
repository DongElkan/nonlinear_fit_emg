cdef:
    void eval_func(double[::1] x, double[::1] y, double[::1] param, double[::1] err, int k)
    void eval_jac(double[::1] x, double[::1] param, double[:, ::1] jac, int k)