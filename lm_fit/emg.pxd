cdef:
    void jacobian_emg(double[::1] x, double a, double u, double s, double b, double[:, ::1] jac)
    void emg(double[::1] x, double[::1] param, double[::1] y)