import unittest

from care.crn.surface import bottom_half_indices
from care.adsorption import place_adsorbate
from care.evaluators.utils import atoms_to_data, extract_adsorbate

from tests import test_inters, surface, co2_from_poscar, surface_from_slab, ammonia_from_poscar

n_slab = len(surface.slab)
num_configs = 3

class TestAdsorbatePlacement(unittest.TestCase):
    def test_placement(self):
        """
        Test that the number of atoms in the adsorption structure is the same as the number of atoms in the intermediate
        """
        for inter in test_inters:
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
                    self.assertEqual(len(structure) , n_slab + n_adsorbate)
                    self.assertTrue(all(inter[i] == structure.get_chemical_symbols().count(i) for i in ["C", "H", "O"]))
                    self.assertTrue(all(i in structure.constraints[0].get_indices() for i in bottom_half_indices(surface.slab)))  # check on constrained bulk atoms
                    self.assertTrue(all(tag == 1 if structure[idx].symbol != "Co" else tag == 0 for idx, tag in enumerate(atoms_tags)))  # check on adsorbate/surface tags
                    self.assertTrue(len(structure) == g.num_nodes)
                    self.assertTrue(all([g.elem[i] == structure[i].symbol for i in range(g.num_nodes)]))  # check on preserved mapping
                    self.assertTrue(gg.num_nodes == n_adsorbate)
                    self.assertTrue(all([gg.elem[i] == structure[gg.idx[i]].symbol for i in range(gg.num_nodes)]))  # check on preserved mapping
                    self.assertEqual(surface.fixed_atoms, list(structure.constraints[0].index))  # check on preserved fixed atoms

    @unittest.skip("Skipping targeted placement tests for user-defined CO2 placement")                
    def test_CO2_placement(self):
        """
        Assert that specific adsorbate placement on specific surface sites works correctly
        This code functionality is not for high-throughput, but for targeted placements.
        """
        adsorptions = place_adsorbate(co2_from_poscar, surface_from_slab, 1, surface_sites=[38,39,40], adsorbate_atom=0)
        self.assertEqual(len(adsorptions), 1)
        graph = atoms_to_data(adsorptions[0], adsorptions[0].get_array("atom_tags"), surface_order=1)
        self.assertTrue(all([x in graph.idx for x in [38,39,40]]))  # check on surface atoms
        adsorbate_anchoring_atom_graph_idx = graph.idx.index(n_slab)  # adsorbate atom used for placement
        self.assertTrue(any([edge[0] == adsorbate_anchoring_atom_graph_idx and edge[1] in [graph.idx.index(i) for i in [38,39,40]] for edge in graph.edge_index.T]))

    def test_NH3_placement(self):
        """
        Assert that specific adsorbate placement on specific surface sites works correctly
        This code functionality is not for high-throughput, but for targeted placements.
        """
        adsorptions = place_adsorbate(ammonia_from_poscar, surface_from_slab, 1, surface_sites=[38,39,40], adsorbate_atom=0)
        self.assertEqual(len(adsorptions), 1)
        graph = atoms_to_data(adsorptions[0], adsorptions[0].get_array("atom_tags"), surface_order=1)
        self.assertTrue(all([x in graph.idx for x in [38,39,40]]))  # check on surface atoms
        adsorbate_anchoring_atom_graph_idx = graph.idx.index(n_slab)  # adsorbate atom used for placement
        self.assertTrue(any([edge[0] == adsorbate_anchoring_atom_graph_idx and edge[1] in [graph.idx.index(i) for i in [38,39,40]] for edge in graph.edge_index.T]))
        with self.assertRaises(ValueError):
            place_adsorbate(ammonia_from_poscar, surface_from_slab, 1, surface_sites=[38,39,1], adsorbate_atom=0)
            place_adsorbate(ammonia_from_poscar, surface_from_slab, 1, surface_sites=[38,39,40], adsorbate_atom=10)
        adsorptions = place_adsorbate(ammonia_from_poscar, surface_from_slab, 3, surface_sites=[43,44,45], adsorbate_atom=0)
        self.assertEqual(len(adsorptions), 3)
        graph = atoms_to_data(adsorptions[0], adsorptions[0].get_array("atom_tags"), surface_order=1)
        self.assertTrue(all([x in graph.idx for x in [43,44,45]]))  # check on surface atoms
        adsorbate_anchoring_atom_graph_idx = graph.idx.index(n_slab)  # adsorbate atom used for placement
        self.assertTrue(any([edge[0] == adsorbate_anchoring_atom_graph_idx and edge[1] in [graph.idx.index(i) for i in [43,44,45]] for edge in graph.edge_index.T]))
