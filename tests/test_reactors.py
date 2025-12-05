import unittest

from ase import Atoms
import numpy as np
from scipy.sparse import csr_matrix

from care import Intermediate
from care.reactors import DifferentialPFR
from care.reactors.differential_pfr import SparsePFR
from care.reactors.utils import analyze_elemental_balance, net_rate

# Test reaction mechanism
# R1) CO(g) + * -> CO*
# R2) O2(g) + 2* -> 2O*
# R3) CO* + O* -> CO2* + *
# R4) CO2* -> CO2(g) + *
# ----------------------
#  CO(g) + 0.5O2(g) -> CO2(g)
# ----------------------

inters = ['CO(g)', 'O2(g)', 'CO2(g)', 'CO*', 'O*', 'CO2*', '*']
gas_mask = np.array([1, 1, 1, 0, 0, 0, 0]).astype(bool)
y0 = np.array([1e6, 3e6, 0.0, 0.5, 0.05, 0.2, 0.25])
pCO, pO2, pCO2, thetaCO, thetaO, thetaCO2, thetastar = y0
intermediates = [Intermediate("CO(g)", Atoms("CO"), is_surface=False, phase="gas"), 
                 Intermediate("O2(g)", Atoms("O2"), is_surface=False, phase="gas"), 
                 Intermediate("CO2(g)", Atoms("CO2"), is_surface=False, phase="gas"), 
                 Intermediate("CO*", Atoms("CO"), is_surface=False, phase="ads"), 
                 Intermediate("O*", Atoms("O"), is_surface=False, phase="ads"), 
                 Intermediate("CO2*", Atoms("CO2"), is_surface=False, phase="ads"), 
                 Intermediate("*", Atoms(), is_surface=True, phase="surf")]
intermediates = {inter.code: inter for inter in intermediates}
inters = {"codes": inters}
for elem in ["C", "H", "O", "N"]:
    inters[elem] = [x[elem] for x in intermediates.values()]
inters["elements"] = ["C", "H", "O", "N"]

v_matrix = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, 2, -1, 0],
        [0, 0, 1, -1],
        [-1, -2, 1, 1],
    ]
)
v_matrix = csr_matrix(v_matrix)
kd = np.array([1e-2, 2e-3, 3e-2, 5e-2])
kr = np.array([1e-4, 1e-5, 1e-1, 1e-1])
k1d, k2d, k3d, k4d = kd[0], kd[1], kd[2], kd[3]
k1r, k2r, k3r, k4r = kr[0], kr[1], kr[2], kr[3]
pfr = DifferentialPFR(v=v_matrix, kd=kd, kr=kr, gas_mask=gas_mask, inters=inters, temperature=500, pressure=1e5, print_progress=False)
rf_correct = np.array([k1d*pCO*thetastar, k2d*pO2*thetastar**2, k3d*thetaCO*thetaO, k4d*thetaCO2])
rb_correct = np.array([k1r*thetaCO, k2r*thetaO**2, k3r*thetaCO2*thetastar, k4r*pCO2*thetastar])
rn_correct = rf_correct - rb_correct
v_forward_correct = csr_matrix(np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1],
                                         [1, 2, 0, 0]])).T
v_backward_correct = csr_matrix(np.array([[0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 1],
                                          [1, 0, 0, 0],
                                          [0, 2, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 1, 1]])).T
dydt0_correct = np.zeros_like(y0)
dydt0_correct[3] = rn_correct[0] - rn_correct[2]  # dCO*dt
dydt0_correct[4] = 2 * rn_correct[1] - rn_correct[2]  # dO*dt
dydt0_correct[5] = rn_correct[2] - rn_correct[3]  # dCO2*dt
dydt0_correct[6] = -rn_correct[0] - 2 * rn_correct[1] + rn_correct[2] + rn_correct[3]  # d*dt
Jy0_correct = np.zeros((7, 7))
# d(dCO*dt)/dx
Jy0_correct[3, 0] = k1d * thetastar  # d(dCO*dt)/dpCO
Jy0_correct[3, 1] = 0.0  # d(dCO*dt)/dpO2
Jy0_correct[3, 2] = 0.0  # d(dCO*dt)/dpCO2
Jy0_correct[3, 3] = -k1r - k3d*thetaO  # d(dCO*dt)/dCO*
Jy0_correct[3, 4] = -k3d*thetaCO  # d(dCO*dt)/dB*
Jy0_correct[3, 5] = k3r*thetastar  # d(dCO*dt)/dC*
Jy0_correct[3, 6] = k1d*pCO + k3r*thetaCO2 # d(dCO*dt)/d*
# d(dO*dt)/dx
Jy0_correct[4, 0] = 0.0  # d(dO*dt)/dpCO
Jy0_correct[4, 1] = 2*k2d*thetastar**2  # d(dO*dt)/dpO2
Jy0_correct[4, 2] = 0.0  # d(dO*dt)/dpCO2
Jy0_correct[4, 3] = -k3d*thetaO  # d(dO*dt)/dCO*
Jy0_correct[4, 4] = -2*2*k2r*thetaO - k3d*thetaCO  # d(dO*dt)/dO*
Jy0_correct[4, 5] = k3r*thetastar  # d(dO*dt)/dCO2*
Jy0_correct[4, 6] = 2*2*k2d*pO2*thetastar + k3r*thetaCO2  # d(dO*dt)/d*
# d(dCO2*dt)/dx
Jy0_correct[5, 0] = 0.0  # d(dCO2*dt)/dpCO
Jy0_correct[5, 1] = 0.0  # d(dCO2*dt)/dpO2
Jy0_correct[5, 2] = k4r*thetastar  # d(dCO2*dt)/dpCO2
Jy0_correct[5, 3] = k3d*thetaO  # d(dCO2*dt)/dCO*
Jy0_correct[5, 4] = k3d*thetaCO  # d(dCO2*dt)/dO*
Jy0_correct[5, 5] = -k3r*thetastar - k4d  # d(dCO2*dt)/dCO2*
Jy0_correct[5, 6] = -k3r*thetaCO2 + k4r*pCO2  # d(dCO2*dt)/d*
# d(d*dt)/dx
Jy0_correct[6, 0] = -k1d*thetastar  # d(d*dt)/dpCO
Jy0_correct[6, 1] = -2*k2d*thetastar**2  # d(d*dt)/dpO2
Jy0_correct[6, 2] = -k4r*thetastar  # d(d*dt)/dpCO2
Jy0_correct[6, 3] = k1r + k3d*thetaO  # d(d*dt)/dCO*
Jy0_correct[6, 4] = 2*2*k2r*thetaO + k3d*thetaCO  # d(d*dt)/dO*
Jy0_correct[6, 5] = -k3r*thetastar + k4d  # d(d*dt)/dCO2*
Jy0_correct[6, 6] = -k1d*pCO - 2*2*k2d*pO2*thetastar - k3r*thetaCO2 - k4r*pCO2  # d(d*dt)/d*

v = v_matrix.T.tocsr()
jvec = lambda arr: SparsePFR.Base.Vector(arr)  # shorthand to call Julia Vector
p = SparsePFR.SparsePFRParams(
    jvec(kd), jvec(kr), SparsePFR.Base.BitVector(gas_mask),
    jvec(v.data.astype('int8')), jvec(v.indices.astype('int64')), jvec(v.indptr.astype('int64')),
    jvec(pfr.v_forward_sparse.data.astype('int8')), jvec(pfr.v_forward_sparse.indices.astype('int64')), jvec(pfr.v_forward_sparse.indptr.astype('int64')),
    jvec(pfr.v_backward_sparse.data.astype('int8')), jvec(pfr.v_backward_sparse.indices.astype('int64')), jvec(pfr.v_backward_sparse.indptr.astype('int64')),
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
        vf_data, vf_indices, vf_indptr = pfr.v_forward_sparse.data, pfr.v_forward_sparse.indices, pfr.v_forward_sparse.indptr
        vb_data, vb_indices, vb_indptr = pfr.v_backward_sparse.data, pfr.v_backward_sparse.indices, pfr.v_backward_sparse.indptr
        rn_numba = net_rate(y0, kd, kr, vf_data, vf_indices, vf_indptr, vb_data, vb_indices, vb_indptr)
        np.testing.assert_allclose(rf, rf_correct, rtol=1e-8, atol=1e-12,
            err_msg=f"Forward rates differ.\nExpected: {rf_correct}\nGot: {rf}")

        np.testing.assert_allclose(rb, rb_correct, rtol=1e-8, atol=1e-12,
            err_msg=f"Backward rates differ.\nExpected: {rb_correct}\nGot: {rb}")

        np.testing.assert_allclose(rn, rn_correct, rtol=1e-8, atol=1e-12,
            err_msg=f"Net rates differ.\nExpected: {rn_correct}\nGot: {rn}")
        
        np.testing.assert_allclose(rn, rn_numba, rtol=1e-8, atol=1e-12,
            err_msg=f"Net rates evaluated with numba and numpy differ.\nExpected: {rn}\nGot: {rn_numba}")

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
        y = pfr.integrate(y0=y0, 
                          solver='Python', 
                          rtol=1e-9, 
                          atol=1e-12, 
                          tfin=1e20)
        balance = analyze_elemental_balance(y, intermediates)
        self.assertTrue(isinstance(y, dict))
        self.assertTrue(y['y'].shape == (7,))
        self.assertTrue(y['forward_rate'].shape == (4,))
        self.assertTrue(y['backward_rate'].shape == (4,))
        self.assertTrue(y['net_rate'].shape == (4,))
        self.assertTrue(y["consumption_rate"].shape == (7,4))
        self.assertTrue(y["total_consumption_rate"].shape == (7,1))
        for elem, ratio in balance.items():
            self.assertAlmostEqual(ratio, 1.0, places=1, msg=f"Elemental balance for {elem} not conserved.")

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
                          rtol=1e-12, 
                          atol=1e-15, 
                          tfin=1e30, 
                          gpu=False, 
                          precision=64, 
                          maxiters=1_000_000)
        balance = analyze_elemental_balance(y, intermediates)
        self.assertTrue(isinstance(y, dict))
        self.assertTrue(y['y'].shape == (7,))
        self.assertTrue(y['forward_rate'].shape == (4,))
        self.assertTrue(y['backward_rate'].shape == (4,))
        self.assertTrue(y['net_rate'].shape == (4,))
        self.assertTrue(y["consumption_rate"].shape == (7,4))
        self.assertTrue(y["total_consumption_rate"].shape == (7,1))
        for elem, ratio in balance.items():
            self.assertAlmostEqual(ratio, 1.0, places=1, msg=f"Elemental balance for {elem} not conserved.")
        y_prec128 = pfr.integrate(y0=y0, 
                          solver='Julia', 
                          rtol=1e-12, 
                          atol=1e-15,
                          tfin=1e30, 
                          gpu=False, 
                          precision=128, 
                          maxiters=1_000_000)
        balance_prec128 = analyze_elemental_balance(y_prec128, intermediates)
        self.assertTrue(isinstance(y_prec128, dict))
        self.assertTrue(y_prec128['y'].shape == (7,))
        self.assertTrue(y_prec128['forward_rate'].shape == (4,))
        self.assertTrue(y_prec128['backward_rate'].shape == (4,))
        self.assertTrue(y_prec128['net_rate'].shape == (4,))
        self.assertTrue(y_prec128["consumption_rate"].shape == (7,4))
        self.assertTrue(y_prec128["total_consumption_rate"].shape == (7,1))
        for elem, ratio in balance_prec128.items():
            self.assertAlmostEqual(ratio, 1.0, places=1, msg=f"Elemental balance for {elem} not conserved.")
        np.testing.assert_allclose(y['y'], y_prec128['y'], rtol=1e-7, atol=1e-3)
