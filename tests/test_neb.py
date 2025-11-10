import unittest

from tests import neb, mlp


class TestNEB(unittest.TestCase):
    def test_1(self):
        self.assertTrue(mlp.is_mlp)
        self.assertEqual(mlp.adsorbate_domain, neb.adsorbate_domain)
        self.assertEqual(mlp.surface_domain, neb.surface_domain)
        self.assertIsInstance(neb.num_images, int)
        self.assertIsInstance(neb.max_steps, int)
        self.assertIsInstance(neb.optimizer, str)
        
