import unittest

import numpy as np

from care import ReactionNetwork
from tests import evaluated_network as nw

mkm = nw.run_microkinetic(iv={"CO": 0.33, "H2": 0.67 }, oc={"T": 450, "P": 1e6}, solver="Julia")

class TestReactionNetwork(unittest.TestCase):
    def test_reaction_network(self):
        self.assertIsInstance(nw, ReactionNetwork)
        nw.temperature = 300
        nw.pressure = 1000000
        self.assertEqual(nw.temperature, 300)
        self.assertEqual(nw.pressure, 1000000)
        self.assertEqual(nw.crn_type, "thermal")
        self.assertGreater(nw.num_closed_shell_mols, 0)
        self.assertGreater(len(nw.adsorptions), 0)

    def test_mkm(self):
        with self.assertRaises(ValueError):
            nw.run_microkinetic(iv={"CH4": 0.5, "H2": 0.5 })
        with self.assertRaises(ValueError):
            ReactionNetwork(oc={"T": 300, "hola": 34})
        self.assertIsInstance(mkm, dict)

    def test_inter_removal(self):
        inter_to_remove_idx = list(nw.intermediates.keys())[3]
        nw.remove_intermediate(nw.intermediates[inter_to_remove_idx])
        self.assertTrue(len(nw.intermediates) > 0)
        self.assertTrue(len(nw.reactions) > 0)

    def test_hubs(self):
        self.assertTrue(len(nw.get_hubs()), len(nw.intermediates))
        self.assertTrue(len(nw.get_hubs(5)), 5)

    def test_mkm_results(self):
        self.assertTrue(len(mkm["reactants_idxs"] == 2))
        self.assertIsInstance(mkm["conversion"], np.ndarray)
        self.assertIsInstance(mkm["selectivity"], dict)
        for elem in nw.elements:
            self.assertTrue(elem in mkm["selectivity"].keys())
            self.assertTrue(elem in mkm["yield"].keys())
            s = mkm["selectivity"][elem]
            self.assertTrue(np.all((s >= 0) & (s <= 1) | np.isnan(s)))
