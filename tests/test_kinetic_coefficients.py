import unittest

from ase import Atoms
import numpy as np

from care import Intermediate
from care.crn.templates import Adsorption, Desorption, BondFormation

inters = [Intermediate("CO(g)", Atoms("CO"), is_surface=False, phase="gas"), 
                 Intermediate("O2(g)", Atoms("O2"), is_surface=False, phase="gas"), 
                 Intermediate("CO2(g)", Atoms("CO2"), is_surface=False, phase="gas"), 
                 Intermediate("CO*", Atoms("CO"), is_surface=False, phase="ads"), 
                 Intermediate("O*", Atoms("O"), is_surface=False, phase="ads"), 
                 Intermediate("CO2*", Atoms("CO2"), is_surface=False, phase="ads"), 
                 Intermediate("*", Atoms(), is_surface=True, phase="surf")]
COg, O2g, CO2g = inters[0], inters[1], inters[2]
COads, Oads, CO2ads, surf = inters[3], inters[4], inters[5], inters[6] 
rxns = [Adsorption([[COg, surf], [COads]], r_type="adsorption"), 
        Adsorption([[O2g, surf], [Oads]], r_type="adsorption"), 
        BondFormation([[COads, Oads], [CO2ads, surf]], r_type="C-O"),
        Desorption([[CO2ads], [CO2g, surf]], r_type="desorption")]

rxns[0].e_rxn, rxns[0].e_act = (-0.5, 0.0), (0.0, 0.0)
rxns[1].e_rxn, rxns[1].e_act = (-0.3, 0.0), (0.0, 0.0)
rxns[2].e_rxn, rxns[2].e_act = (-0.2, 0.0), (0.6, 0.0)
rxns[3].e_rxn, rxns[3].e_act = (0.9, 0.0), (0.9, 0.0)

# kinetic and thermo constants
T= 400
keq0 = 0.0
keq1 = 0.0
keq2 = 0.0
keq3 = 0.0

kdir0 = 0.0
krev0 = 0.0
kdir1 = 0.0
krev1 = 0.0
kdir2 = 0.0
krev2 = 0.0
kdir3 = 0.0
krev3 = 0.0

class TestKineticCoefficients(unittest.TestCase):

    def test_adsorption(self):
        self.assertTrue(rxns[0].adsorbate == COg)
        self.assertTrue(rxns[1].adsorbate == O2g)
        self.assertTrue(rxns[3].adsorbate == CO2g)
        self.assertAlmostEqual(rxns[0].adsorbate_mass, 28.0, delta=0.05)
        self.assertAlmostEqual(rxns[1].adsorbate_mass, 32.0, delta=0.05)
        self.assertAlmostEqual(rxns[3].adsorbate_mass, 44.0, delta=0.05)
        



