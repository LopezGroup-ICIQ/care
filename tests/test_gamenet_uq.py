import random
import unittest

from dask.distributed import Client, LocalCluster

from care import gen_blueprint
from care.evaluators import load_surface
from care.evaluators.gamenet_uq import GameNetUQInter, METALS, METAL_STRUCT_DICT, FACET_DICT

intermediates, _ = gen_blueprint(1, 1, False, False, False)
surface = load_surface(metal="Pt", hkl="111")
model_inter = GameNetUQInter(surface, num_configs=2)
test_inters_seq = random.sample(list(intermediates.values()), 3)
test_inters_mp = list(intermediates.values())

class TestEvaluator(unittest.TestCase):

    def test_surface(self):
        """Check that all surfaces have active sites
        """
        for metal in METALS:
            for facet in FACET_DICT[METAL_STRUCT_DICT[metal]]:
                surface = load_surface(metal=metal, hkl=facet)
                assert surface.num_atoms != 0
                assert surface.vacuum_height >= 10.0

    def test_model(self):
        assert model_inter.model.parameters() != None

    def test_serial_eval(self):        
        for inter in test_inters_seq:
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
        futures = client.map(f, test_inters_mp)
        results = client.gather(futures)
        for inter in results:
            if inter.phase == "ads":
                assert len(inter.ads_configs) == 2
                self.assertIsInstance(inter.ads_configs["0"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["0"]["s"], float)
                self.assertIsInstance(inter.ads_configs["1"]["mu"], float)
                self.assertIsInstance(inter.ads_configs["1"]["s"], float)
                self.assertGreater(inter.ads_configs["0"]["s"], 0.0)
                self.assertGreater(inter.ads_configs["1"]["s"], 0.0)
            elif inter.phase in ("gas", "surf"):
                assert len(inter.ads_configs) == 1
