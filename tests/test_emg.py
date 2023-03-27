import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize

from lm_fit import calculate_emg, estimate_emg_p0, calculate_jacobian_emg

from test_nl_fit import load_curve


def _ls_func(x, p1, p2, p3, p4):
    params = np.fromiter([p1, p2, p3, p4], np.float64)
    return calculate_emg(x, params)


def _load_curve_data():
    y = load_curve()
    n = y.size
    x = np.arange(n) * 0.03 + 0.01
    y -= y.min()
    return x, y


def _sp_curve_fit(x, y):
    param = estimate_emg_p0(x, y)
    popt, _ = optimize.curve_fit(_ls_func, x, y, p0=np.asarray(param))
    return popt


def test_calculate_emg():
    x, y = _load_curve_data()
    sp_param = _sp_curve_fit(x, y)
    yp = calculate_emg(x, sp_param)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y)
    ax.plot(x, yp, "--", c="firebrick")
    plt.show()


def test_emg_jacobian():
    x, y = _load_curve_data()
    sp_param = _sp_curve_fit(x, y)

    jac = calculate_jacobian_emg(x, sp_param)
    print(sp_param)

    i = 3

    yp0 = calculate_emg(x, sp_param)
    sp_param_move = sp_param.copy()
    sp_param_move[i] *= 1.0000001
    yp1 = calculate_emg(x, sp_param_move)

    diff_a = (yp1 - yp0) / (sp_param[i] * 0.0000001)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(jac[:, i], diff_a, ".")
    # ax.plot(diff_a, ".")
    plt.show()


if __name__ == "__main__":
    # test_calculate_emg()
    test_emg_jacobian()
