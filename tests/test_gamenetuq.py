import unittest
import pytest

from care.crn.intermediate import AdsorbedSpecies, GasSpecies, SurfaceSite
from tests.shared_data import test_inters


class TestEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from care.evaluators.gamenet_uq import GameNetUQevaluator
            cls.model_inter = GameNetUQevaluator(num_configs=2)
        except ImportError:
            pytest.skip("GAME-Net-UQ not installed, skipping these tests.")

    def test_model(self):
        assert self.model_inter.model.parameters() != None

    def test_serial_eval(self):        
        for inter in test_inters:
            self.model_inter(inter)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
            elif isinstance(GasSpecies, SurfaceSite):
                assert inter.E != None

    def test_parallel_eval(self):
        from dask.distributed import Client, LocalCluster
        
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
        client = Client(address=cluster)
        
        def f(inter):
            if '_worker_model' not in globals():
                from care.evaluators.gamenet_uq import GameNetUQevaluator
                globals()['_worker_model'] = GameNetUQevaluator(num_configs=2)

            globals()['_worker_model'](inter)
            return inter
            
        futures = client.map(f, test_inters)
        results = client.gather(futures)
        
        for inter in results:
            self.assertTrue(inter.is_evaluated)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["0"]["s"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["s"], float)
                self.assertGreater(inter.ads_configs["0"]["s"], 0.0)
                self.assertGreater(inter.ads_configs["1"]["s"], 0.0)
            elif isinstance(inter, (GasSpecies, SurfaceSite)):
                assert getattr(inter, 'E', None) is not None
                
        client.close()
        cluster.close()
