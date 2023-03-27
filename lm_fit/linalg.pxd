cdef:
    int qr(double[:, ::1] a, int[::1] piv_index, double[::1] beta)
    void solve_linear(double[:, ::1] a, double[::1] z, double[::1] x)
    double norm(double[::1] x)
    void col_norm(double[:, ::1] x, double[::1] c)
    void row_norm(double[:, ::1] x, double[::1] c)