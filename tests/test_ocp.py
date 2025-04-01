import random
import unittest

from dask.distributed import Client, LocalCluster

from care import gen_blueprint
from care.evaluators import load_surface
from care.evaluators.ocp import OCPIntermediateEvaluator

import pytest


intermediates, rxns = gen_blueprint(1, 1, False, False, False)
surface = load_surface(metal="Fe", hkl="110")
model_inter = OCPIntermediateEvaluator(surface, num_configs=2, max_steps=3)
test_inters = random.sample(list(intermediates.values()), 3)

class TestEvaluator(unittest.TestCase):
    @pytest.mark.skip(reason="Failing only on GitHub Actions")
    def test_serial_eval(self):
        for inter in test_inters:
            model_inter(inter)
            if inter.phase == "ads":
                assert len(inter.ads_configs) == 2
            elif inter.phase in ("gas", "surf"):
                assert len(inter.ads_configs) == 1

    def test_parallel_eval(self):
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
        client = Client(address=cluster)
        def f(inter):
            print(inter.code + "\n")
            model_inter(inter)
            return inter
        futures = client.map(f, test_inters)
        results = client.gather(futures)
        for inter in results:
            if inter.phase == "ads":
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["0"]["s"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["s"], float)
                self.assertAlmostEqual(inter.ads_configs["0"]["s"], 0.0, places=3)
                self.assertAlmostEqual(inter.ads_configs["1"]["s"], 0.0, places=3)
            elif inter.phase in ("gas", "surf"):
                assert len(inter.ads_configs) == 1
