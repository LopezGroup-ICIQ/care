import unittest

import numpy as np

from care.crn.reactors import DifferentialPFR


v_matrix = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [-1, -1, 1, 1],
    ]
)
kd, kr = np.ones(4), np.ones(4)
gas_mask = np.array([1, 1, 1, 0, 0, 0, 0])
inters = ['A(g)', 'B(g)', 'C(g)', 'A*', 'B*', 'C*', '*']
pfr = DifferentialPFR(v_matrix, kd, kr, gas_mask, inters, temperature=500, pressure=1e5)

class TestDifferentialPFR(unittest.TestCase):

    def test_stoic_forward(self):
        """
        Check that the forward stoichiometric matrix is correctly implemented
        """
        self.assertTrue(np.all(pfr.v_forward_dense >= 0))
        self.assertTrue(np.array_equal(pfr.v_forward_dense, np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1],
                                                        [1, 1, 0, 0]]).T))
        self.assertEqual(pfr.v_forward_dense.shape, (4, 7))

    def test_stoic_backward(self):
        """
        Check that the backward stoichiometric matrix is correctly implemented
        """
        self.assertTrue(np.all(pfr.v_backward_dense >= 0))
        self.assertTrue(np.array_equal(pfr.v_backward_dense, np.array([[0, 0, 0, 0],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 1],
                                                         [1, 0, 0, 0],
                                                         [0, 1, 0, 0],
                                                         [0, 0, 1, 0],
                                                         [0, 0, 1, 1]]).T))
        self.assertEqual(pfr.v_backward_dense.shape, (4, 7))

    def test_rates(self):
        """
        Check that the rates are correctly implemented
        """
        # define y as numpy array with values between 0 and 1
        y = np.random.rand(7)
        rf = pfr.forward_rate(y)
        rb = pfr.backward_rate(y)
        rn = pfr.net_rate(y)
        self.assertTrue(np.all(pfr.forward_rate(y) >= 0))
        self.assertEqual(rf.shape, (4,))
        self.assertTrue(np.all(pfr.backward_rate(y) >= 0))
        self.assertEqual(rb.shape, (4,))
        self.assertEqual(rn.shape, (4,))

    def test_ode_gasmask(self):
        """
        Check that the gas mask is correctly implemented
        """
        # define y as numpy array with values between 0 and 1
        y = np.random.rand(7)
        dydt = pfr.ode(0, y)
        dydt_42 = pfr.ode(42, y)
        self.assertTrue(np.array_equal(dydt_42, dydt))
        self.assertEqual(dydt[0], 0.0)
        self.assertEqual(dydt[1], 0.0)
        # self.assertEqual(dydt[2], 0.0)
