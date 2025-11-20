import unittest
from tests import surface, surface_from_bulk, surface_from_slab

from ase import Atoms
import numpy as np

from care.crn.surface import parse_hkl_string, bottom_half_indices


class TestSurface(unittest.TestCase):
    def test_hkl(self):
        x = ["111", "0001", "10m10", "10m11", "2m1m12"]
        y = [(1, 1, 1), (0, 0, 1), (1, 0, 0), (1, 0, 1), (2, -1, 2)]
        hkl_dict = dict(zip(x,y)) 
        for x, y  in hkl_dict.items():
            self.assertEqual(parse_hkl_string(x), y)

    def test_attrs(self):
        self.assertIsInstance(surface.slab, Atoms)
        self.assertIsInstance(surface.num_layers, int)
        self.assertIsInstance(surface.slab_height, float)
        self.assertIsInstance(surface.vacuum_height, float)
        self.assertIsInstance(surface.area, float)
        atoms_tags = list(surface.slab.get_array("atom_tags"))
        self.assertTrue(all(tag == 0 for tag in atoms_tags))
        self.assertIsInstance(bottom_half_indices(surface.slab), np.ndarray)
        self.assertIsInstance(surface.fixed_atoms, list)

    def test_from_bulk_poscar(self):
        self.assertIsInstance(surface_from_bulk.slab, Atoms)
        self.assertIsNotNone(surface_from_bulk.facet)
        self.assertAlmostEqual(surface.vacuum_height, 15.0, delta=1.5)
        self.assertIsInstance(surface.fixed_atoms, list)

    def test_from_slab_poscar(self):
        self.assertIsInstance(surface_from_slab.slab, Atoms)
        self.assertIsNone(surface_from_slab.facet)
        self.assertIsInstance(surface.fixed_atoms, list)