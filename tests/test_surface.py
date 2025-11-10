import unittest
from tests import surface

from ase import Atoms

from care.crn.surface import parse_hkl_string


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