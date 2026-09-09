import unittest
import pytest

from care.crn.intermediate import SurfaceSite, AdsorbedSpecies, GasSpecies
from tests.shared_data import test_inters


class TestEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from care.evaluators.sevennet import SevenNetevaluator
            cls.mlip_dispersion = SevenNetevaluator(model="7net-omni", 
                                         modal="mpa", 
                                         num_configs=2, 
                                         max_steps=3,
                                         dispersion=True,
                                         device="cuda", 
                                         enable_cueq=False)
            cls.mlip_no_dispersion = SevenNetevaluator(model="7net-omni",
                                         modal="mpa",
                                         num_configs=2,
                                         max_steps=3,
                                         dispersion=False,
                                         device="cuda",
                                         enable_cueq=False)
        except ImportError:
            pytest.skip("SevenNet not installed, skipping these tests.")
    
    def test_serial_eval(self):       
        for inter in test_inters:
            self.mlip_dispersion(inter)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
            elif isinstance(inter, (SurfaceSite, GasSpecies)):
                assert inter.E != None
            self.mlip_no_dispersion(inter)
