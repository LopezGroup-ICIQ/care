import unittest

import numpy as np

from care import ReactionNetwork
from tests import evaluated_network as nw

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
        self.assertTrue(nw.is_evaluated)

    def test_inter_removal(self):
        inter_to_remove_idx = list(nw.intermediates.keys())[3]
        nw.remove_intermediate(nw.intermediates[inter_to_remove_idx])
        self.assertTrue(len(nw.intermediates) > 0)
        self.assertTrue(len(nw.reactions) > 0)

    def test_hubs(self):
        self.assertTrue(len(nw.get_hubs()), len(nw.intermediates))
        self.assertTrue(len(nw.get_hubs(5)), 5)

