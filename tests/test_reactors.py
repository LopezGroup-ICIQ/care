import pytest
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from care.reactors import DifferentialPFR
from care.reactors.differential_pfr import SparsePFR


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
v_matrix = csr_matrix(v_matrix)
kd = np.array([1e-2, 2e-3, 3e-2, 5e-2])
kr = np.array([1e-4, 1e-5, 1e-1, 1e-1])
gas_mask = np.array([1, 1, 1, 0, 0, 0, 0]).astype(bool)
inters = ['A(g)', 'B(g)', 'C(g)', 'A*', 'B*', 'C*', '*']
pfr = DifferentialPFR(v=v_matrix, kd=kd, kr=kr, gas_mask=gas_mask, inters=inters, temperature=500, pressure=1e5)
y0 = np.array([1e6, 3e6, 0.0, 0.5, 0.05, 0.2, 0.25])
rf_correct = np.array([1e-2*1e6*0.25, 2e-3*3e6*0.25, 3e-2*0.5*0.05, 5e-2*0.2])
rb_correct = np.array([1e-4*0.5, 1e-5*0.05, 1e-1*0.2*0.25, 0.0])
rn_correct = rf_correct - rb_correct
v_forward_correct = csr_matrix(np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1],
                                         [1, 1, 0, 0]])).T
v_backward_correct = csr_matrix(np.array([[0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 1],
                                          [1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 1, 1]])).T
dydt0_correct = np.zeros_like(y0)
dydt0_correct[3] = 1*(1e-2*1e6*0.25 - 1e-4*0.5) - 1*(3e-2*0.5*0.05 - 1e-1*0.2*0.25)
dydt0_correct[4] = 1*(2e-3*3e6*0.25 - 1e-5*0.05) - 1*(3e-2*0.5*0.05 - 1e-1*0.2*0.25)
dydt0_correct[5] = 1*(3e-2*0.5*0.05 - 1e-1*0.2*0.25) - 1*(5e-2*0.2 - 0.0)
dydt0_correct[6] = -1*(1e-2*1e6*0.25 - 1e-4*0.5) -1*(2e-3*3e6*0.25 - 1e-5*0.05) + 1*(3e-2*0.5*0.05 - 1e-1*0.2*0.25) + 1*(5e-2*0.2 - 0.0)

class TestDifferentialPFR(unittest.TestCase):

    def test_stoic_forward(self):
        """
        Check that the forward stoichiometric matrix is correctly implemented
        """
        self.assertTrue(np.all(pfr.v_forward_sparse.data >= 0))
        self.assertTrue((pfr.v_forward_sparse != v_forward_correct).nnz == 0)
        self.assertEqual(pfr.v_forward_sparse.shape, (4, 7))

    def test_stoic_backward(self):
        """
        Check that the backward stoichiometric matrix is correctly implemented
        """
        self.assertTrue(np.all(pfr.v_backward_sparse.data >= 0))
        self.assertTrue((pfr.v_backward_sparse != v_backward_correct).nnz == 0)
        self.assertEqual(pfr.v_backward_sparse.shape, (4, 7))

    def test_rates(self):
        """
        Check that the rates are correctly implemented
        """
        rf = pfr.forward_rate(y0)
        rb = pfr.backward_rate(y0)
        rn = pfr.net_rate(y0)
        self.assertTrue(np.array_equal(rf, rf_correct))
        self.assertTrue(np.array_equal(rb, rb_correct))
        self.assertTrue(np.array_equal(rn, rn_correct))
        self.assertEqual(rf.shape, (4,))
        self.assertEqual(rb.shape, (4,))
        self.assertEqual(rn.shape, (4,))

    def test_ode_gasmask(self):
        """
        Check that the gas mask is correctly implemented, 
        the ODE is autonomous and returns the expected values.
        """
        dydt0 = pfr.ode(0, y0)
        dydt0_42 = pfr.ode(42, y0)
        self.assertTrue(np.array_equal(dydt0_42, dydt0))
        self.assertTrue(np.array_equal(dydt0, dydt0_correct))

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

    def test_ode_jl(self):
        v = v_matrix.T.tocsr()
        p = SparsePFR.SparsePFRParams(
            kd, kr, gas_mask,
            v.data, v.indices, v.indptr,
            pfr.v_forward_sparse.data, pfr.v_forward_sparse.indices, pfr.v_forward_sparse.indptr,
            pfr.v_backward_sparse.data, pfr.v_backward_sparse.indices, pfr.v_backward_sparse.indptr,
        )
        net_rates = SparsePFR.sparse_net_rate(y0, p)
        dydt0 = np.zeros_like(y0)
        SparsePFR.ode_pfr_b(dydt0, y0, p, 0.0)
        self.assertTrue(np.array_equal(net_rates, rn_correct))
        self.assertTrue(np.array_equal(dydt0, dydt0_correct))

    def test_integration_jl_cpu(self):
        """
        Check that the integration with Julia is correctly implemented.
        """
        y = pfr.integrate(y0=y0, 
                          solver='Julia', 
                          rtol=1e-6, 
                          atol=1e-12, 
                          sstol=1e-7, 
                          tfin=1e6, 
                          gpu=False)
        self.assertTrue(isinstance(y, dict))
        self.assertTrue(y['y'].shape == (7,))
        self.assertTrue(y['forward_rate'].shape == (4,))
        self.assertTrue(y['backward_rate'].shape == (4,))
        self.assertTrue(y['net_rate'].shape == (4,))

    # @pytest.mark.skip(reason="GPU unavailable on GitHub Actions")
    # def test_integration_jl_gpu(self):
    #     """
    #     Check that the integration with Julia is correctly implemented
    #     when integration is performed on GPU.
    #     """
    #     y = pfr.integrate(y0=y0, 
    #                       solver='Julia', 
    #                       rtol=1e-6, 
    #                       atol=1e-12, 
    #                       sstol=1e-7, 
    #                       tfin=1e6, 
    #                       gpu=True)
    #     self.assertTrue(isinstance(y, dict))
    #     self.assertTrue(y['y'].shape == (7,))
    #     self.assertTrue(y['forward_rate'].shape == (4,))
    #     self.assertTrue(y['backward_rate'].shape == (4,))
    #     self.assertTrue(y['net_rate'].shape == (4,))
