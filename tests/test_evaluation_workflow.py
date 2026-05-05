import unittest
from ase import Atoms
from care import gen_blueprint, Intermediate

class TestReactionNetwork(unittest.TestCase):
    def test_reaction_network_unified_intermediates(self):
        crn = gen_blueprint(1, 2, False, False, False)
        self.assertGreater(len(crn.reactions), 0, "Blueprint generated 0 reactions.")
        self.assertGreater(len(crn.nodes), 0, "Blueprint generated 0 nodes.")
        phases_modified = set()
        for node in crn.nodes:
            if isinstance(node, Intermediate):
                phases_modified.add(node.phase)
                if node.phase == "ads":
                    node.ads_configs['0'] = {'ase': Atoms("H"), 'mu': -1.0, 's': 0.0}
                elif node.phase == "gas":
                    node.ads_configs['gas'] = {'ase': Atoms("H"), 'mu': -2.0, 's': 0.0}
                elif node.phase == "surf":
                    node.ads_configs['gas'] = {'ase': Atoms("H"), 'mu': -3.0, 's': 0.0}
        self.assertIn("ads", phases_modified, "No adsorbed species found in blueprint.")
        self.assertIn("gas", phases_modified, "No gas species found in blueprint.")
        self.assertIn("surf", phases_modified, "No surface site found in blueprint.")

        for rxn in crn.reactions:
            for inter in list(rxn.reactants) + list(rxn.products):
                self.assertEqual(
                    len(inter.ads_configs), 1, 
                    f"Species {inter.code} ({inter.phase}) has empty ads_configs."
                )
                if inter.phase == "ads":
                    self.assertIn('0', inter.ads_configs)
                    self.assertEqual(inter.ads_configs['0']['mu'], -1.0)
                elif inter.phase == "gas":
                    self.assertIn('gas', inter.ads_configs)
                    self.assertEqual(inter.ads_configs['gas']['mu'], -2.0)
                elif inter.phase == "surf":
                    self.assertIn('gas', inter.ads_configs)
                    self.assertEqual(inter.ads_configs['gas']['mu'], -3.0)