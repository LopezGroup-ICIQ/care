import unittest

from care import gen_blueprint, load_surface
from care.crn.surface import bottom_half_indices
from care.adsorption import place_adsorbate
from care.evaluators.utils import atoms_to_data, extract_adsorbate

intermediates, _ = gen_blueprint(1, 1, False, False, False)
surface = load_surface(metal="Cu", hkl="111")
n_slab = len(surface.slab)
num_configs = 5

class TestAdsorbatePlacement(unittest.TestCase):
    def test_placement(self):
        """
        Test that the number of atoms in the adsorption structure is the same as the number of atoms in the intermediate
        """
        for inter in intermediates.values():
            if inter.phase == "ads":
                n_adsorbate = len(inter.molecule)
                adsorptions1 = place_adsorbate(inter, surface, num_configs)
                adsorptions2 = place_adsorbate(inter, surface, num_configs)
                self.assertTrue(len(adsorptions1) == num_configs)
                self.assertTrue(adsorptions1 == adsorptions2, msg="DockOnSurf returns DIFFERENT ORDER EVERYTIME IS CALLED!!!")
                for structure in adsorptions1:
                    atoms_tags = list(structure.get_array("atom_tags"))
                    g = atoms_to_data(structure, atoms_tags)
                    gg = extract_adsorbate(g, atoms_tags)
                    self.assertTrue(all(inter[i] == structure.get_chemical_symbols().count(i) for i in ["C", "H", "O"]))
                    self.assertTrue(all(i in structure.constraints[0].get_indices() for i in bottom_half_indices(surface.slab)))  # check on constrained bulk atoms
                    self.assertTrue(all(tag == 1 if structure[idx].symbol != "Cu" else tag == 0 for idx, tag in enumerate(atoms_tags)))  # check on adsorbate/surface tags
                    self.assertTrue(len(structure) == g.num_nodes)
                    self.assertTrue(all([g.elem[i] == structure[i].symbol for i in range(g.num_nodes)]))  # check on preserved mapping
                    self.assertTrue(gg.num_nodes == n_adsorbate)
                    self.assertTrue(all([gg.elem[i] == structure[gg.idx[i]].symbol for i in range(gg.num_nodes)]))  # check on preserved mapping

