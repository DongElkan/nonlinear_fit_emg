from lm_fit import test_qrsolve, test_qr, test_solve_linear
import numpy as np

import matplotlib.pyplot as plt


def test_qrsolve_():
    n = 20
    x1 = np.linspace(0.1, 5.1, num=n) + np.random.randn(n) / 5
    x2 = np.linspace(1, 10, num=20) * 0.2 + 5 + np.random.randn(n)
    x3 = np.sqrt(np.linspace(1, 200, num=n)) + np.random.randn(n) * 1.2
    x4 = np.exp(np.linspace(0.1, 3, num=n)) + np.random.randn(n)

    b = np.linspace(1, 10, num=n) + np.random.randn(n) / 3
    b.astype(np.float64)

    # fig, ax = plt.subplots()
    # ax.plot(x1)
    # ax.plot(x2)
    # ax.plot(x3)
    # ax.plot(x4)
    # plt.show()

    a = np.c_[x1, x2, x3, x4, np.random.randn(n, 5)]
    a.astype(np.float64)
    p = a.shape[1]

    ra, piv, beta = test_qr(a.copy())
    x_0 = test_solve_linear(a.copy(), b.copy())
    with np.printoptions(suppress=True, precision=4):
        print(x_0)
        print(np.dot(a, x_0))

    # fig, ax = plt.subplots()
    # ax.plot(np.dot(a, x_0), b, ".", ms=8.)
    # x0, x1 = ax.get_xlim()
    # y0, y1 = ax.get_ylim()
    # ax.plot([x0, x1], [x0, x1], c="firebrick")
    # ax.set_xlim((x0, x1))
    # ax.set_ylim((y0, y1))
    # plt.show()

    r = np.zeros((p, p), dtype=np.float64)
    d = np.zeros(p, dtype=np.float64)
    for i in range(p):
        r[i][i:] = ra[i][i:]
        d[i] = ra[i][i]

    qtb = np.zeros(p, dtype=np.float64)
    b_x = b.copy()
    for j in range(p):
        t = ((ra[j + 1:, j] * b_x[j + 1:]).sum() + b_x[j]) * beta[j]
        b_x[j] -= t
        b_x[j + 1:] -= t * ra[j + 1:, j]
        qtb[j] = b_x[j]

    x_t = np.zeros(p, dtype=np.float64)
    for i in range(1, p + 1):
        j = p - i
        s = (ra[j, j + 1:] * x_t[j + 1:]).sum()
        x_t[j] = (b_x[j] - s) / ra[j, j]

    for i in range(1, p + 1):
        j = p - i
        if piv[j] != j:
            k = piv[j]
            t = x_t[k]
            x_t[k] = x_t[j]
            x_t[j] = t

    fig, ax = plt.subplots()
    ax.plot(np.dot(a, x_t), b, ".", ms=8.)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.plot([x0, x1], [x0, x1], c="firebrick")
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    plt.show()

    x = np.zeros(p, dtype=np.float64)
    sdiag = np.zeros(p, dtype=np.float64)
    test_qrsolve(r, d, qtb, x, sdiag)

    with np.printoptions(suppress=True, precision=4):
        print(np.dot(a, x))
        print(b)


if __name__ == "__main__":
    test_qrsolve_()
