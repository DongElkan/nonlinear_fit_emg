from lm_fit import nl_fit, estimate_emg_p0, calculate_emg
import numpy as np

import matplotlib.pyplot as plt


def load_curve():
    return np.genfromtxt(r"curve.txt")


def test_nl_fit():
    y = load_curve()
    n = y.size
    x = np.arange(n) * 0.03 + 0.01
    param = estimate_emg_p0(x, y)
    print(np.asarray(param))

    nl_fit(x, y, param, 1)
    print(np.asarray(param))

    yp = calculate_emg(x, param)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y)
    ax.plot(x, yp, "--", c="firebrick")
    plt.show()


if __name__ == "__main__":
    test_nl_fit()
