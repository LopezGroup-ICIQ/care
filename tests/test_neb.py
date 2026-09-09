import unittest

from care.evaluators import MACEevaluator
from care.crn.utils.graph import is_adsorbate_fragmented, adsorption_filter
from care.crn.templates.dissociation import BondBreaking, _build_fragment_nx
from networkx import is_connected

from tests import evaluated_network

mlp = MACEevaluator(size="small", max_steps=5, device="cuda", allow_shared_calculator=False, parallel=False)

reaction = None
for rxn in evaluated_network.reactions:
    if isinstance(rxn, BondBreaking):
        reaction = rxn
        break


class TestNEB(unittest.TestCase):
    def test_1(self):
        self.assertTrue(mlp.is_mlp)
        self.assertIsInstance(mlp.num_images, int)
        self.assertIsInstance(mlp.max_steps, int)
        self.assertIsInstance(mlp.optimizer, str)

    def test_2(self):
        bc_graph = _build_fragment_nx(reaction.stoic, list(reaction.products))
        reaction.get_states(mlp)
        self.assertFalse(is_connected(bc_graph))
        self.assertFalse(is_adsorbate_fragmented(reaction.is_graph))
        self.assertTrue(is_adsorbate_fragmented(reaction.fs_graph))
        self.assertTrue(adsorption_filter(reaction.fs_graph))
        self.assertTrue(adsorption_filter(reaction.is_graph))
        mlp(reaction)
        self.assertIsNotNone(reaction.neb_images)
        self.assertIsNotNone(reaction.neb_energies)
        self.assertIsInstance(reaction.neb_energies[2], float)
        self.assertEqual(len(reaction.neb_images), mlp.num_images+2)
        
