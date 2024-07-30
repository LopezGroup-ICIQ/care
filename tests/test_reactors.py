import unittest

import numpy as np

from care.crn.reactors import DifferentialPFR


# Test reaction network
# R1) A(g) + * -> A*
# R2) B(g) + * -> B*
# R3) A* + B* -> C* + *
# R4) C* -> C(g) + *
# ----------------------
#  A(g) + B(g) -> C(g)
# ----------------------


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
gas_mask = np.array([1, 1, 1, 0, 0, 0, 0]).astype(bool)
inters = ['A(g)', 'B(g)', 'C(g)', 'A*', 'B*', 'C*', '*']
pfr = DifferentialPFR(v_matrix, kd, kr, gas_mask, inters, temperature=500, pressure=1e5)
y0 = np.array([1e6, 1e6, 0, 0, 0, 0, 1])

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
        y = np.random.rand(7)
        dydt = pfr.ode(0, y)
        dydt_42 = pfr.ode(42, y)
        self.assertTrue(np.array_equal(dydt_42, dydt))  # ODE is autonomous
        self.assertEqual(dydt[0], 0.0)
        self.assertEqual(dydt[1], 0.0)
        self.assertEqual(dydt[2], 0.0)

    def test_integration_scipy(self):
        """
        Check that the integration with scipy is correctly implemented
        """
        y = pfr.integrate(y0=y0, solver='Python', rtol=1e-6, atol=1e-12, sstol=1e-7, tfin=1e6)
        self.assertTrue(isinstance(y, dict))
        self.assertTrue(y['y'].shape == (7,))  
        self.assertTrue(y['forward_rate'].shape == (4,))
        self.assertTrue(y['backward_rate'].shape == (4,))
        self.assertTrue(y['net_rate'].shape == (4,))

    def test_integration_jl_cpu(self):
        """ 
        Check that the integration with Julia is correctly implemented
        """
        y = pfr.integrate(y0=y0, solver='Julia', rtol=1e-6, atol=1e-12, sstol=1e-7, tfin=1e6)
        self.assertTrue(isinstance(y, dict))
        self.assertTrue(y['y'].shape == (7,))
        self.assertTrue(y['forward_rate'].shape == (4,))
        self.assertTrue(y['backward_rate'].shape == (4,))
        self.assertTrue(y['net_rate'].shape == (4,))

    def test_integration_jl_gpu(self):
        """        
        Check that the integration with Julia is correctly implemented
        when integration is performed on GPU.
        """
        pass
