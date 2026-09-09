import unittest
import pytest

from tests.shared_data import test_inters
from care.crn.intermediate import AdsorbedSpecies, GasSpecies, SurfaceSite

class TestEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from care.evaluators.upet import UPETevaluator
            cls.model_inter = UPETevaluator(num_configs=2, max_steps=2)
        except ImportError:
            pytest.skip("UPET not installed, skipping these tests.")

    def test_serial_eval(self):
        for inter in test_inters:
            self.model_inter(inter)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
            else:
                assert inter.E != None

    def test_parallel_eval(self):
        from dask.distributed import Client, LocalCluster
        
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
        client = Client(address=cluster)
        
        def f(inter):
            if '_worker_model' not in globals():
                from care.evaluators.upet import UPETevaluator
                globals()['_worker_model'] = UPETevaluator(num_configs=2, max_steps=2)

            globals()['_worker_model'](inter)
            return inter
            
        futures = client.map(f, test_inters)
        results = client.gather(futures)
        
        for inter in results:
            self.assertTrue(inter.is_evaluated)
            if isinstance(inter, AdsorbedSpecies):
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
            elif isinstance(inter, (GasSpecies, SurfaceSite)):
                assert getattr(inter, 'E', None) is not None
                
        client.close()
        cluster.close()
