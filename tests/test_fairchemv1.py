import pytest
import unittest

from dask.distributed import Client, LocalCluster

from care.crn.intermediate import AdsorbedSpecies, GasSpecies, SurfaceSite
from tests import test_inters


class TestEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from care.evaluators.fairchemv1 import FairChemV1evaluator
            cls.model_inter = FairChemV1evaluator(
                num_configs=2, max_steps=2
            )
        except ImportError:
            pytest.skip("Fairchem-core v1 not installed, skipping these tests.")

    def test_serial_eval(self):
        for inter in test_inters:
            self.model_inter(inter)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
            elif isinstance(GasSpecies, SurfaceSite):
                assert inter.E != None

    def test_parallel_eval(self):
        cluster = LocalCluster(n_workers=2, threads_per_worker=1)
        client = Client(address=cluster)
        model = self.model_inter 
        def f(inter):
            print(inter.code + "\n")
            model(inter)
            return inter
        futures = client.map(f, test_inters)
        results = client.gather(futures)
        for inter in results:
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
            elif isinstance(GasSpecies, SurfaceSite):
                assert inter.E != None
