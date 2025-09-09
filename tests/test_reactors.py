import pytest
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from care.reactors import DifferentialPFR
from care.reactors.differential_pfr import SparsePFR


# Test reaction mechanism
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
k1d, k2d, k3d, k4d = kd[0], kd[1], kd[2], kd[3]
k1r, k2r, k3r, k4r = kr[0], kr[1], kr[2], kr[3]
gas_mask = np.array([1, 1, 1, 0, 0, 0, 0]).astype(bool)
inters = ['A(g)', 'B(g)', 'C(g)', 'A*', 'B*', 'C*', '*']
pfr = DifferentialPFR(v=v_matrix, kd=kd, kr=kr, gas_mask=gas_mask, inters=inters, temperature=500, pressure=1e5)
y0 = np.array([1e6, 3e6, 0.0, 0.5, 0.05, 0.2, 0.25])
pA, pB, pC, thetaA, thetaB, thetaC, theta_star = y0
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
Jy0_correct = np.zeros((7, 7))
# d(dA*dt)/dx
Jy0_correct[3, 0] = k1d * theta_star  # d(dA*dt)/dA(g)
Jy0_correct[3, 1] = 0.0  # d(dA*dt)/dB(g)
Jy0_correct[3, 2] = 0.0  # d(dA*dt)/dC(g)
Jy0_correct[3, 3] = -k1r - k3d*thetaB  # d(dA*dt)/dA*
Jy0_correct[3, 4] = -k3d*thetaA  # d(dA*dt)/dB*
Jy0_correct[3, 5] = k3r*theta_star  # d(dA*dt)/dC*
Jy0_correct[3, 6] = k1d*pA + k3r*thetaC # d(dA*dt)/d*
# d(dB*dt)/dx
Jy0_correct[4, 0] = 0.0  # d(dB*dt)/dA(g)
Jy0_correct[4, 1] = k2d*theta_star  # d(dB*dt)/dB(g)
Jy0_correct[4, 2] = 0.0  # d(dB*dt)/dC(g)
Jy0_correct[4, 3] = -k3d*thetaB  # d(dB*dt)/dA*
Jy0_correct[4, 4] = -k2r - k3d*thetaA  # d(dB*dt)/dB*
Jy0_correct[4, 5] = k3r*theta_star  # d(dB*dt)/dC*
Jy0_correct[4, 6] = k2d*pB + k3r*thetaC  # d(dB*dt)/d*
# d(dC*dt)/dx
Jy0_correct[5, 0] = 0.0  # d(dC*dt)/dA(g)
Jy0_correct[5, 1] = 0.0  # d(dC*dt)/dB(g)
Jy0_correct[5, 2] = k4r*theta_star  # d(dC*dt)/dC(g)
Jy0_correct[5, 3] = k3d*thetaB  # d(dC*dt)/dA*
Jy0_correct[5, 4] = k3d*thetaA  # d(dC*dt)/dB*
Jy0_correct[5, 5] = -k3r*theta_star - k4d  # d(dC*dt)/dC*
Jy0_correct[5, 6] = -k3r*thetaC + k4r*pC  # d(dC*dt)/d*
# d(d*dt)/dx
Jy0_correct[6, 0] = -k1d*theta_star  # d(d*dt)/dA(g)
Jy0_correct[6, 1] = -k2d*theta_star  # d(d*dt)/dB(g)
Jy0_correct[6, 2] = -k4r*theta_star  # d(d*dt)/dC(g)
Jy0_correct[6, 3] = k1r + k3d*thetaB  # d(d*dt)/dA*
Jy0_correct[6, 4] = k2r + k3d*thetaA  # d(d*dt)/dB*
Jy0_correct[6, 5] = -k3r*theta_star + k4d  # d(d*dt)/dC*
Jy0_correct[6, 6] = -k1d*pA - k2d*pB - k3r*thetaC - k4r*pC  # d(d*dt)/d*

v = v_matrix.T.tocsr()
p = SparsePFR.SparsePFRParams(
    kd, kr, gas_mask,
    v.data, v.indices, v.indptr,
    pfr.v_forward_sparse.data, pfr.v_forward_sparse.indices, pfr.v_forward_sparse.indptr,
    pfr.v_backward_sparse.data, pfr.v_backward_sparse.indices, pfr.v_backward_sparse.indptr,
)

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

    def test_rates_py(self):
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

    def test_rates_jl(self):
        net_rates = SparsePFR.sparse_net_rate(y0, p)
        self.assertTrue(np.array_equal(net_rates, rn_correct))

    def test_ode_gasmask(self):
        """
        Check that the gas mask is correctly implemented, 
        the ODE is autonomous and returns the expected values.
        """
        dydt0 = pfr.ode(0, y0)
        dydt0_42 = pfr.ode(42, y0)
        self.assertTrue(np.array_equal(dydt0_42, dydt0))
        self.assertTrue(np.array_equal(dydt0, dydt0_correct))

    def test_jacobian_py(self):
        """
        Check that the Jacobian is correctly implemented in Python
        """
        Jy0 = pfr.jacobian(0, y0)
        self.assertTrue(np.array_equal(Jy0.toarray(), Jy0_correct))
        self.assertEqual(Jy0.shape, (7, 7))

    def test_jacobian_jl(self):
        jacobian = np.array(SparsePFR.sparse_jacobian_outplace(y0, p))
        np.testing.assert_allclose(jacobian, Jy0_correct, rtol=1e-8, atol=1e-12)

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
        dydt0 = np.zeros_like(y0)
        SparsePFR.ode_pfr_b(dydt0, y0, p, 0.0)
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
