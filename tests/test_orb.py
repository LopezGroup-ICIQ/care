import pytest
import unittest

from dask.distributed import Client, LocalCluster

from care.crn.intermediate import SurfaceSite, AdsorbedSpecies, GasSpecies
from tests import test_inters


class TestEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from care.evaluators.orb import ORBevaluator
            cls.mlip = ORBevaluator(version="orb-v3-conservative-inf-omat", num_configs=2, max_steps=3, device="cuda")
        except ImportError:
            pytest.skip("Orb not installed, skipping these tests.")
    
    def test_serial_eval(self):       
        for inter in test_inters:
            self.mlip(inter)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
            elif isinstance(inter, (SurfaceSite, GasSpecies)):
                assert inter.E != None

    def test_parallel_eval(self):
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
        client = Client(address=cluster)
        model = self.mlip
        def f(inter):
            model(inter)
            return inter
        futures = client.map(f, test_inters)
        results = client.gather(futures)
        for inter in results:
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
            elif isinstance(inter, (SurfaceSite, GasSpecies)):
                assert inter.E != None
