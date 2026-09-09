import unittest
import pytest

from dask.distributed import Client, LocalCluster

from care.crn.intermediate import SurfaceSite, AdsorbedSpecies, GasSpecies
from tests import test_inters


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

    # def test_parallel_eval(self):
    #     cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    #     client = Client(address=cluster)
    #     model = self.mlip
    #     def f(inter):
    #         model(inter)
    #         return inter
    #     futures = client.map(f, test_inters)
    #     results = client.gather(futures)
    #     for inter in results:
    #         if isinstance(inter, AdsorbedSpecies):
    #             assert len(inter.ads_configs) == 2
    #             self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
    #             self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
    #         elif isinstance(inter, (SurfaceSite, GasSpecies)):
    #             assert inter.E != None