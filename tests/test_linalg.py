import unittest
import numpy as np

from lm_fit import test_qr, test_solve_linear


class TestLinalg(unittest.TestCase):
    def test_qr_decom(self):
        a = np.random.randn(20, 5).astype(np.float64)
        a0 = a.copy()
        a, piv, b = test_qr(a)
        q, r = np.linalg.qr(a0)

        with np.printoptions(suppress=True, precision=4):
            print(a[:5, :5])
            print(r)

        b2 = np.zeros(5)
        for j in range(5):
            b2[j] = 2. / (np.linalg.norm(a[j+1:, j]) ** 2 + 1.)
        self.assertTrue(np.allclose(b, b2))

    def test_solve_linear_x(self):
        a = np.c_[np.ones((20, 1)), np.random.randn(20, 4).astype(np.float64)]
        x = np.fromiter([2., 4., 1., 1., 3.], np.float64)
        z = np.dot(a, x)
        z += np.random.randn(20) * np.linalg.norm(z) * 0.01
        a0 = a.copy()
        z0 = z.copy()
        x1 = test_solve_linear(a, z)
        x2, res, rnk, vals = np.linalg.lstsq(a0, z0, rcond=None)
        r1 = np.linalg.norm(np.dot(a0, x1) - z0)

        self.assertTrue(abs(r1 ** 2 - res[0]) <= 1e-6)


if __name__ == "__main__":
    unittest.main()
