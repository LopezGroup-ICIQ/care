import unittest
import os
import pytest
from tests.shared_data import surface, surface_from_bulk, surface_from_slab

from ase import Atoms
from ase.build import bulk
import numpy as np

from care.crn.surface import parse_hkl_string, bottom_half_indices, Surface


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

    def test_surface_from_mp(self):
        api_key = os.environ.get("MP_API_KEY")
        
        if not api_key:
            pytest.skip("MP_API_KEY environment variable not set. Skipping Materials Project test.")
        surf = Surface.from_mp(
            mp_id="mp-30",
            hkl="111",
            mp_api_key=api_key
        )
        slab_elements = set([x.symbol for x in surf.slab])
        self.assertTrue("Cu" in slab_elements)
        self.assertTrue(len(slab_elements)==1)

    def test_slab_diag(self):
        atoms = Atoms('Cu', cell=[(3, 0, 0), (0, 4, 0), (0, 0, 10)], pbc=True)
        surf = Surface(ase_atoms_slab=atoms, facet="111")
        self.assertEqual(surf.slab_diag, 5.0)

    def test_from_bulk_poscar_with_float_num_layers_and_hkl_str(self):
        """Test generation using float num_layers (height-based) and hkl as a string."""
        dummy_bulk = bulk('Cu', 'fcc', a=3.6)
        target_height = 4.0
        surf = Surface.from_bulk_poscar(
            bulk_poscar_path=dummy_bulk, 
            hkl="100", 
            num_layers=target_height
        )        
        self.assertEqual(surf.facet, "100")
        self.assertGreater(surf.slab_height, target_height)

    def test_from_bulk_poscar_invalid_hkl_list(self):
        dummy_bulk = bulk('Cu', 'fcc', a=3.6)
        
        with self.assertRaisesRegex(ValueError, "Miller index hkl must be a string or a list of 3 integers."):
            Surface.from_bulk_poscar(bulk_poscar_path=dummy_bulk, hkl=[1, 1], num_layers=3)
            
        with self.assertRaisesRegex(ValueError, "Miller index hkl must be a string or a list of 3 integers."):
            Surface.from_bulk_poscar(bulk_poscar_path=dummy_bulk, hkl=[1, 1, 1.5], num_layers=3)

    def test_from_bulk_poscar_invalid_num_layers(self):
        """Test that passing a non-int/float to num_layers raises an error."""
        dummy_bulk = bulk('Cu', 'fcc', a=3.6)
        
        with self.assertRaisesRegex(ValueError, "num_layers must be an int or float."):
            Surface.from_bulk_poscar(bulk_poscar_path=dummy_bulk, hkl="111", num_layers="three")