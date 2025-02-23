import random
import unittest

from care import gen_blueprint
from care.evaluators import load_surface
from care.adsorption import place_adsorbate

intermediates, _ = gen_blueprint(1, 1, False, False, False)
surface = load_surface(metal="Pt", hkl="111")
num_configs = 5


class TestAdsorbatePlacement(unittest.TestCase):
    def test_placement(self):
        """
        Test that the number of atoms in the adsorption structure is the same as the number of atoms in the intermediate
        """
        test_inters = random.sample(list(intermediates.values()), 5)
        for inter in test_inters:
            adsorptions = place_adsorbate(inter, surface, num_configs)
            self.assertTrue(len(adsorptions) == num_configs)
            for structure in adsorptions:
                self.assertTrue(inter['C'] == structure.get_chemical_symbols().count('C'))
                self.assertTrue(inter['H'] == structure.get_chemical_symbols().count('H'))
                self.assertTrue(inter['O'] == structure.get_chemical_symbols().count('O'))
