import unittest

from care.evaluators import NEBReactionEnergyEstimator, MACEIntermediateEvaluator, FairChemV2IntermediateEvaluator
from care.evaluators.utils import is_adsorbate_fragmented, adsorption_filter
from care.crn.templates import BondBreaking
from networkx import is_connected

from tests import evaluated_network, surface

mlp = MACEIntermediateEvaluator(surface=surface, size="small", max_steps=5)
# mlp = FairChemV2IntermediateEvaluator(surface=surface, max_steps=3, device="cuda")

neb = NEBReactionEnergyEstimator(mlp=mlp, max_steps=10)

# choose random reaction from evaluated network of type BondBreaking for testing
reaction = None
for rxn in evaluated_network.reactions:
    if isinstance(rxn, BondBreaking):
        reaction = rxn
        break


class TestNEB(unittest.TestCase):
    def test_1(self):
        self.assertTrue(mlp.is_mlp)
        self.assertEqual(mlp.adsorbate_domain, neb.adsorbate_domain)
        self.assertEqual(mlp.surface_domain, neb.surface_domain)
        self.assertIsInstance(neb.num_images, int)
        self.assertIsInstance(neb.max_steps, int)
        self.assertIsInstance(neb.optimizer, str)

    def test_2(self):
        bc_graph = neb._build_product_nx(reaction)
        neb.get_fs(reaction)
        self.assertFalse(is_connected(bc_graph))
        self.assertFalse(is_adsorbate_fragmented(reaction.is_graph))
        self.assertTrue(is_adsorbate_fragmented(reaction.fs_graph))
        self.assertTrue(adsorption_filter(reaction.fs_graph))
        self.assertTrue(adsorption_filter(reaction.is_graph))
        neb.run_neb(reaction)
        self.assertIsNotNone(reaction.neb_images)
        self.assertIsNotNone(reaction.neb_energies)
        self.assertEqual(len(reaction.neb_images), neb.num_images+2)
        
