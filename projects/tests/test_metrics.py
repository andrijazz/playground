import unittest
import numpy as np
import utils.metrics as metrics


class TestMetrics(unittest.TestCase):

    def test_rse(self):
        p = np.arange(24).reshape((2, 3, 4))
        gt = np.arange(24).reshape((2, 3, 4)) * 2
        out = metrics.rse(p, gt)
        self.assertListEqual(np.floor(out).tolist(), [75, 733])


if __name__ == '__main__':
    unittest.main()
