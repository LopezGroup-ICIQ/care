import unittest
from ase import Atoms
from care import Intermediate

from care.crn.utils.blueprint import gen_blueprint
from care.crn.intermediate import AdsorbedSpecies, GasSpecies, SurfaceSite

class TestReactionNetwork(unittest.TestCase):
    def test_reaction_network_unified_intermediates(self):
        crn = gen_blueprint(ncc=1, noc=2)
        self.assertGreater(len(crn.reactions), 0, "Blueprint generated 0 reactions.")
        self.assertGreater(len(crn.nodes), 0, "Blueprint generated 0 nodes.")
        phases_modified = set()
        for node in crn.nodes:
            if isinstance(node, Intermediate):
                phases_modified.add(node.phase)
                if isinstance(node, AdsorbedSpecies):
                    node.ads_configs['0'] = {'ase': Atoms("H"), 'mu': -1.0}
                elif isinstance(node, GasSpecies):
                    node.E = -2.0
                elif isinstance(node, SurfaceSite):
                    node.E = -3.0
        self.assertIn("ads", phases_modified, "No adsorbed species found in blueprint.")
        self.assertIn("gas", phases_modified, "No gas species found in blueprint.")
        self.assertIn("surf", phases_modified, "No surface site found in blueprint.")

        for rxn in crn.reactions:
            for inter in list(rxn.reactants) + list(rxn.products):
                if isinstance(inter, AdsorbedSpecies):
                    self.assertIn('0', inter.ads_configs)
                    self.assertEqual(inter.ads_configs['0']['mu'], -1.0)
                elif isinstance(inter, GasSpecies):
                    self.assertEqual(inter.E, -2.0)
                elif isinstance(inter, SurfaceSite):
                    self.assertEqual(inter.E, -3.0)