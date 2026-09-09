import os
import unittest
import tempfile
from copy import deepcopy

from ase import Atoms
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from care import ReactionNetwork
from care.crn.intermediate import GasSpecies, AdsorbedSpecies, SurfaceSite
from care.reactors import DifferentialPFR
from care.reactors.utils import net_rate, generate_simulation_report, MKMRun
from care.crn.templates import Adsorption, Desorption, BondFormation

# Test reaction mechanism
# R1) CO(g) + * -> CO*
# R2) O2(g) + 2* -> 2O*
# R3) CO* + O* -> CO2* + *
# R4) CO2* -> CO2(g) + *
# ----------------------
#  CO(g) + 0.5O2(g) -> CO2(g)
# ----------------------

T = 500  # K
P = 1e5  # Pa
ATOL = 1e-12  # -
RTOL = 1e-9  # -
TFIN = 1e20  # s
inters = ['CO(g)', 'CO*', 'CO2(g)', 'CO2*', 'O*', 'O2(g)', '*']
gas_mask_correct = np.array([1, 0, 1, 0, 0, 1, 0]).astype(bool)
y0 = np.array([1e6, 0.5, 0.0, 0.2, 0.05, 3e6, 0.25])
pCO, thetaCO, pCO2, thetaCO2, thetaO, pO2, thetastar = y0
x = [GasSpecies("CO(g)", Atoms("CO")), 
     GasSpecies("O2(g)", Atoms("O2")), 
     GasSpecies("CO2(g)", Atoms("CO2")), 
     AdsorbedSpecies("CO*", Atoms("CO")), 
     AdsorbedSpecies("O*", Atoms("O")), 
     AdsorbedSpecies("CO2*", Atoms("CO2")), 
     SurfaceSite()]
COg, O2g, CO2g, COstar, Ostar, CO2star, star = x[0], x[1], x[2], x[3], x[4], x[5], x[6]  

reactions = [
    Adsorption([[COg, star], [COstar]]), 
    Adsorption([[O2g, star], [Ostar]]), 
    BondFormation([[COstar, Ostar], [CO2star, star]]), 
    Desorption([[CO2star], [CO2g, star]])
]
crn = ReactionNetwork(reactions)

v_matrix_correct = np.array(
    [
        [-1, 0, 0, 0],
        [1, 0, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 2, -1, 0],
        [0, -1, 0, 0],
        [-1, -2, 1, 1],
    ]
)

v_matrix_reversed_correct = np.array(
    [
        [-1, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, -1, -1],
        [0, 2, 1, 0],
        [0, -1, 0, 0],
        [-1, -2, -1, 1],
    ]
) # same mechanism, R3 written in opposite direction

v_matrix_reversed_full_correct = np.array(
    [
        [1, 0, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, -1],
        [0, 0, -1, 1],
        [0, -2, 1, 0],
        [0, 1, 0, 0],
        [1, 2, -1, -1],
    ]
) # same mechanism, ALL reactions rewritten in opposite direction
v_matrix_correct = csr_matrix(v_matrix_correct)
v_matrix_reversed_correct = csr_matrix(v_matrix_reversed_correct) 
v_matrix_reversed_full_correct = csr_matrix(v_matrix_reversed_full_correct) 
kd = np.array([1e-2, 2e-3, 3e-2, 5e-2])
kr = np.array([1e-4, 1e-5, 1e-1, 1e-1])
k1d, k2d, k3d, k4d = kd[0], kd[1], kd[2], kd[3]
k1r, k2r, k3r, k4r = kr[0], kr[1], kr[2], kr[3]

# PFR base model
pfr = DifferentialPFR(crn, T=T, P=P, print_progress=False)
pfr.kd = kd
pfr.kr = kr

# PFR with all reversed reactions in the network
crn_full_reversed = deepcopy(crn)
for i, _ in enumerate(crn_full_reversed.reactions):
    crn_full_reversed.reverse_reaction(i)
pfr_full_reversed = DifferentialPFR(crn_full_reversed, T=T, P=P, print_progress=False)
pfr_full_reversed.kd = kr
pfr_full_reversed.kr = kd

# PFR with only R3 reversed
kd_r3_reversed = np.array([1e-2, 2e-3, 1e-1, 5e-2])
kr_r3_reversed = np.array([1e-4, 1e-5, 3e-2, 1e-1])
crn_r3_reversed = deepcopy(crn)
crn_r3_reversed.reverse_reaction(2)
pfr_r3_reversed = DifferentialPFR(crn_r3_reversed, T=T, P=P, print_progress=False)
pfr_r3_reversed.kd = kd_r3_reversed
pfr_r3_reversed.kr = kr_r3_reversed


rf_correct = np.array([k1d*pCO*thetastar, k2d*pO2*thetastar**2, k3d*thetaCO*thetaO, k4d*thetaCO2])
rb_correct = np.array([k1r*thetaCO, k2r*thetaO**2, k3r*thetaCO2*thetastar, k4r*pCO2*thetastar])
rn_correct = rf_correct - rb_correct
v_forward_correct = csr_matrix(np.array([[1, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 1, 0, 0],
                                         [1, 2, 0, 0]])).T

v_backward_correct = csr_matrix(np.array([[0, 0, 0, 0],
                                          [1, 0, 0, 0],
                                          [0, 0, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 2, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 1, 1]])).T

# Correct ODE system
dydt0_correct = np.zeros_like(y0)
dydt0_correct[1] = rn_correct[0] - rn_correct[2]  # dCO*dt
dydt0_correct[3] = rn_correct[2] - rn_correct[3]  # dCO2*dt
dydt0_correct[4] = 2 * rn_correct[1] - rn_correct[2]  # dO*dt
dydt0_correct[6] = -rn_correct[0] - 2 * rn_correct[1] + rn_correct[2] + rn_correct[3]  # d*dt

#Correct Jacobian of ODE system
Jy0_correct = np.zeros((7, 7))
# d(dCO*dt)/dx
Jy0_correct[1, 0] = k1d * thetastar  # d(dCO*dt)/dpCO
Jy0_correct[1, 1] = -k1r - k3d*thetaO  # d(dCO*dt)/dCO*
Jy0_correct[1, 2] = 0.0  # d(dCO*dt)/dpCO2
Jy0_correct[1, 3] = k3r*thetastar  # d(dCO*dt)/dCO2*
Jy0_correct[1, 4] = -k3d*thetaCO  # d(dCO*dt)/dO*
Jy0_correct[1, 5] = 0.0  # d(dCO*dt)/dpO2
Jy0_correct[1, 6] = k1d*pCO + k3r*thetaCO2 # d(dCO*dt)/d*

# d(dCO2*dt)/dx
Jy0_correct[3, 0] = 0.0  # d(dCO2*dt)/dpCO
Jy0_correct[3, 1] = k3d*thetaO  # d(dCO2*dt)/dCO*
Jy0_correct[3, 2] = k4r*thetastar  # d(dCO2*dt)/dpCO2
Jy0_correct[3, 3] = -k3r*thetastar - k4d  # d(dCO2*dt)/dCO2*
Jy0_correct[3, 4] = k3d*thetaCO  # d(dCO2*dt)/dO*
Jy0_correct[3, 5] = 0.0  # d(dCO2*dt)/dpO2
Jy0_correct[3, 6] = -k3r*thetaCO2 + k4r*pCO2  # d(dCO2*dt)/d*

# d(dO*dt)/dx
Jy0_correct[4, 0] = 0.0  # d(dO*dt)/dpCO
Jy0_correct[4, 1] = -k3d*thetaO  # d(dO*dt)/dCO*
Jy0_correct[4, 2] = 0.0  # d(dO*dt)/dpCO2
Jy0_correct[4, 3] = k3r*thetastar  # d(dO*dt)/dCO2*
Jy0_correct[4, 4] = -2*2*k2r*thetaO - k3d*thetaCO  # d(dO*dt)/dO*
Jy0_correct[4, 5] = 2*k2d*thetastar**2  # d(dO*dt)/dpO2
Jy0_correct[4, 6] = 2*2*k2d*pO2*thetastar + k3r*thetaCO2  # d(dO*dt)/d*

# d(d*dt)/dx
Jy0_correct[6, 0] = -k1d*thetastar  # d(d*dt)/dpCO
Jy0_correct[6, 1] = k1r + k3d*thetaO  # d(d*dt)/dCO*
Jy0_correct[6, 2] = -k4r*thetastar  # d(d*dt)/dpCO2
Jy0_correct[6, 3] = -k3r*thetastar + k4d  # d(d*dt)/dCO2*
Jy0_correct[6, 4] = 2*2*k2r*thetaO + k3d*thetaCO  # d(d*dt)/dO*
Jy0_correct[6, 5] = -2*k2d*thetastar**2  # d(d*dt)/dpO2
Jy0_correct[6, 6] = -k1d*pCO - 2*2*k2d*pO2*thetastar - k3r*thetaCO2 - k4r*pCO2  # d(d*dt)/d*

v = v_matrix_correct.T.tocsr()
SparsePFR = pfr._get_julia_solver() 
import juliacall
jl_base = juliacall.Main.Base 
jvec = lambda arr: jl_base.Vector(arr)

p = SparsePFR.SparsePFRParams(
    jvec(kd), jvec(kr), jl_base.BitVector(pfr.gas_mask),
    jvec(v.data.astype('int8')), jvec(v.indices.astype('int64')), jvec(v.indptr.astype('int64')),
    jvec(pfr.v_forward_sparse.data.astype('int8')), jvec(pfr.v_forward_sparse.indices.astype('int64')), jvec(pfr.v_forward_sparse.indptr.astype('int64')),
    jvec(pfr.v_backward_sparse.data.astype('int8')), jvec(pfr.v_backward_sparse.indices.astype('int64')), jvec(pfr.v_backward_sparse.indptr.astype('int64')),
)

def set_crn_energetics(crn):
    """
    Set raw IS, FS, and TS energies.
    """
    crn[0].e_is = 0.0
    crn[1].e_is = 0.0
    crn[2].e_is = 0.0
    crn[3].e_is = 0.0

    crn[0].e_ts = None
    crn[1].e_ts = None
    crn[2].e_ts = 0.4
    crn[3].e_ts = None

    crn[0].e_fs = -0.5
    crn[1].e_fs = -0.4
    crn[2].e_fs = -0.1
    crn[3].e_fs = 0.5

class TestDifferentialPFR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output = pfr.integrate(y0=y0, 
                                   solver='Python', 
                                   rtol=RTOL, 
                                   atol=ATOL, 
                                   tfin=TFIN)
        cls.crn = crn
        
    def test_types(self):
        self.assertIsInstance(self.output, dict)
        self.assertTrue(len(crn.global_reactions) == 1)

    def test_stoic_forward(self):
        """
        Check that the forward stoichiometric matrix is correctly implemented
        """
        pfr._reset_state()
        self.assertTrue(np.all(pfr.v_forward_sparse.data >= 0))
        self.assertTrue((pfr.v_forward_sparse != v_forward_correct).nnz == 0)
        self.assertEqual(pfr.v_forward_sparse.shape, (4, 7))

    def test_stoic_backward(self):
        """
        Check that the backward stoichiometric matrix is correctly implemented
        """
        pfr._reset_state()
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

    def test_gas_mask(self):
        np.testing.assert_equal(pfr.gas_mask, gas_mask_correct)
        self.assertTrue(len(pfr.gas_mask) == 7)
        self.assertTrue(np.count_nonzero(pfr.gas_mask), 3)
        self.assertTrue(np.sum(pfr.gas_mask == 0) == 4)

    def test_ode_is_autonomous(self):
        """
        Check that the ODE is autonomous.
        """
        self.assertTrue(np.array_equal(pfr.ode(42, y0), pfr.ode(0, y0)))

    def test_dydt_is_correct(self):
        dydt = pfr.ode(0, y0)
        np.testing.assert_array_equal(np.sort(dydt), np.sort(dydt0_correct))

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

    def test_species_order(self):
        pfr._reset_state()
        self.assertEqual(pfr.elements, ["C", "O"])
        self.assertEqual(pfr.inters_formula, ['CO', 'CO', 'CO2', 'CO2', 'O', 'O2', '*'])
        self.assertEqual(pfr.inters_info["C"], [1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(pfr.inters_info["O"], [1, 1, 2, 2, 1, 2, 0])

    def test_integration_scipy(self):
        """
        Check that the integration with scipy is correctly implemented
        """
        self.assertTrue(self.output['y'].shape == (7,))
        self.assertTrue(self.output['forward_rate'].shape == (4,))
        self.assertTrue(self.output['backward_rate'].shape == (4,))
        self.assertTrue(self.output['net_rate'].shape == (4,))
        self.assertTrue(self.output["formation_rate"].shape == (7,4))
        self.assertTrue(self.output["total_formation_rate"].shape == (7,1))

    def test_integration_reversed_reactions(self):
        """
        Check two scenarios: (I) that the results of the microkinetic simulations do not change if surface elementary
        reactions (all except adsorption and desorption) are reversed in direction. (II) results of the microkinetic 
        simulations do not change if ALL reactions (including adsorption and desorption) are reversed in direction; 
        second scenario must be extended in the future when setting energy values rather than directly kinetic coeffs.
        """
        output_rev_partial = pfr_r3_reversed.integrate(y0=y0, 
                          solver='Python', 
                          rtol=RTOL, 
                          atol=ATOL, 
                          tfin=TFIN)
        output_rev_full = pfr_full_reversed.integrate(y0=y0, 
                          solver='Python', 
                          rtol=RTOL, 
                          atol=ATOL, 
                          tfin=TFIN)
        np.testing.assert_allclose(self.output['y'], output_rev_partial['y'], rtol=1e-7, atol=1e-3)
        np.testing.assert_allclose(self.output['y'], output_rev_full['y'], rtol=1e-7, atol=1e-3)

    def test_ode_jl(self):
        dydt0 = np.zeros_like(y0)
        SparsePFR.ode_pfr_b(dydt0, y0, p, 0.0)
        self.assertTrue(np.array_equal(dydt0, dydt0_correct))

    def test_integration_jl_cpu(self):
        """
        Check that the integration with Julia is correctly implemented and that 
        the solver with higher precision works.
        """
        output_jl = pfr.integrate(y0=y0, 
                          solver='Julia', 
                          rtol=1e-12, 
                          atol=1e-15, 
                          tfin=1e30, 
                          precision=64, 
                          maxiters=1_000_000)
        self.assertTrue(output_jl['y'].shape == (7,))
        self.assertTrue(output_jl['forward_rate'].shape == (4,))
        self.assertTrue(output_jl['backward_rate'].shape == (4,))
        self.assertTrue(output_jl['net_rate'].shape == (4,))
        self.assertTrue(output_jl["formation_rate"].shape == (7,4))
        self.assertTrue(output_jl["total_formation_rate"].shape == (7,1))

        output_jl_prec128 = pfr.integrate(y0=y0, 
                          solver='Julia', 
                          rtol=1e-12, 
                          atol=1e-15,
                          tfin=1e30, 
                          precision=128, 
                          maxiters=1_000_000)
        self.assertTrue(output_jl_prec128['y'].shape == (7,))
        self.assertTrue(output_jl_prec128['forward_rate'].shape == (4,))
        self.assertTrue(output_jl_prec128['backward_rate'].shape == (4,))
        self.assertTrue(output_jl_prec128['net_rate'].shape == (4,))
        self.assertTrue(output_jl_prec128["formation_rate"].shape == (7,4))
        self.assertTrue(output_jl_prec128["total_formation_rate"].shape == (7,1))

        np.testing.assert_allclose(output_jl['y'], output_jl_prec128['y'], rtol=1e-7, atol=1e-3)

    def test_run_base_and_balances(self):
        """Test the basic run integration, ensuring performance metrics and balances exist."""
        set_crn_energetics(self.crn)
        iv = {"CO": 0.5, "O2": 0.5}
        results = pfr.run(iv=iv, rtol=RTOL, atol=ATOL, tfin=TFIN)
        self.assertIsInstance(results, MKMRun)
        self.assertIsNotNone(results.y0)
        self.assertIsNotNone(results.y)
        self.assertIsNotNone(results.conversion)
        self.assertIsNotNone(results.selectivity)
        self.assertIsNotNone(results.yyield)

        self.assertIsInstance(results.y, np.ndarray)
        self.assertIsInstance(results.total_formation_rate, np.ndarray)
        self.assertIsInstance(results.reactants_idxs, np.ndarray)
        self.assertIsInstance(results.products_idxs, np.ndarray)
        self.assertIsInstance(results.conversion, np.ndarray)
        self.assertIsInstance(results.selectivity, dict)
        self.assertIsInstance(results.yyield, dict)

        self.assertEqual(results.nsims, 1)
        self.assertTrue(len(results.reactants_idxs), 2)
        self.assertTrue(len(results.products_idxs), 1)
        self.assertIn(0, results.reactants_idxs)
        self.assertIn(5, results.reactants_idxs)
        self.assertIn(2, results.products_idxs)

        for idx in results.reactants_idxs:
            self.assertLessEqual(results.total_formation_rate[idx], 
                                 0.0, 
                                 f"Reactant {inters[idx]} is being produced rather than consumed!")
        for idx in results.products_idxs:
            self.assertGreaterEqual(results.total_formation_rate[idx], 
                                    0.0, 
                                    f"Product {inters[idx]} is being consumed rather than produced!")


        for elem in ["C", "O", "*"]:
            self.assertAlmostEqual(
                results.balances[elem], 
                1.0, 
                places=1, 
                msg=f"Elemental balance for {elem} not respected."
            )

    def test_run_cov0_poisoning(self):
        """Test initialization with a heavily poisoned surface."""
        set_crn_energetics(self.crn)
        iv = {"CO": 0.8, "O2": 0.2}
        cov0 = {"CO": 0.9}
        
        results = pfr.run(iv=iv, cov0=cov0, rtol=RTOL, atol=ATOL, tfin=TFIN)
        self.assertIn(0.9, results.y0)
        self.assertIsNotNone(results.y)
        self.assertEqual(results.status, 1)

    def test_run_eapp(self):
        """Test apparent activation energy (E_app) estimation via temperature perturbation."""
        set_crn_energetics(self.crn)
        iv = {"CO": 0.5, "O2": 0.5}
        results = pfr.run(
            iv=iv, eapp=True, dT=5.0, rtol=RTOL, atol=ATOL, tfin=TFIN
        )
        self.assertIsNotNone(results.eapp)
        self.assertTrue(len(results.eapp) > 0)
        for eapp in results.eapp.values():
            self.assertGreaterEqual(eapp, 0.0)
        self.assertEqual(results.nsims, 3)


    def test_run_napp(self):
        """Test apparent reaction order (n_app) estimation, verifying inert injection."""
        set_crn_energetics(self.crn)
        iv = {"CO": 0.4, "O2": 0.4, "Ar": 0.2}
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = pfr.run(
                iv=iv, napp=True, dy=0.02, rtol=RTOL, atol=ATOL, tfin=TFIN
            )
            self.assertTrue(any("Treating as inert" in str(warn.message) for warn in w))
            
        self.assertIsNotNone(results.napp)
        self.assertTrue(len(results.napp) > 0)
        self.assertEqual(results.nsims, 5)

    def test_run_drc(self):
        """Test Degree of Rate Control (DRC) estimation across all elementary steps."""
        set_crn_energetics(self.crn)
        iv = {"CO": 0.5, "O2": 0.5}
        results = pfr.run(
            iv=iv, drc=True, de=0.001, rtol=RTOL, atol=ATOL, tfin=TFIN
        )
        
        self.assertIsNotNone(results.drc)
        self.assertEqual(len(results.raw_drc), pfr.nr)
        self.assertEqual(results.nsims, 6)  # based on provided energy profile in set_energetics()
        for p_idx in results.products_idxs:
            drc_sum = sum(drc_array[p_idx] for drc_array in results.raw_drc.values() if not np.isnan(drc_array[p_idx]))
            self.assertAlmostEqual(drc_sum, 1.0, delta=0.01, msg=f"Sum of DRCs for product index {p_idx} is not ~1.0")
            self.assertIsInstance(results.rds, dict)
            self.assertTrue(all(isinstance(rds, tuple) and len(rds) == 3 for rds in results.rds.values()))
            self.assertTrue(all(isinstance(rds[0], int) and isinstance(rds[1], str) and isinstance(rds[2], float) for rds in results.rds.values()))

    def test_report_generation(self):
        """Verify that the Excel file is created with the correct sheets."""
        set_crn_energetics(self.crn)
        test_pfr = deepcopy(pfr)
        full_run = test_pfr.run(iv={"CO": 0.5, "O2": 0.5}, rtol=RTOL, atol=ATOL, tfin=TFIN, rewire_network=False)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_path = os.path.join(tmpdirname, "test_report.xlsx")
            generate_simulation_report(full_run.to_dict(), output_filename=output_path)
            self.assertTrue(os.path.exists(output_path))
            
            excel_file = pd.ExcelFile(output_path)
            expected_sheets = {'Species', 'Reactions', 'Activity', 'Settings'}
            self.assertTrue(expected_sheets.issubset(set(excel_file.sheet_names)))
            
            df_species = pd.read_excel(output_path, sheet_name='Species', engine='openpyxl')
            self.assertIn("InChIKey", df_species.columns)
            self.assertTrue(len(df_species) >= 7)

            df_reactions = pd.read_excel(output_path, sheet_name='Reactions', engine='openpyxl')
            self.assertIn("net rate (1/s)", df_reactions.columns)
            self.assertEqual(len(df_reactions), 4)
