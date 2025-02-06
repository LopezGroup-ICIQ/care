import random
import unittest

from care import gen_blueprint
from care.evaluators import load_surface
from care.evaluators.gamenet_uq import GameNetUQInter, METALS, METAL_STRUCT_DICT, FACET_DICT

intermediates, _ = gen_blueprint(1, 1, False, False, False)
surface = load_surface(metal="Pt", hkl="111")
model_inter = GameNetUQInter(surface, num_configs=2)

class TestEvaluator(unittest.TestCase):

    def test_surface(self):
        """Check that all surfaces have active sites
        """
        for metal in METALS:
            for facet in FACET_DICT[METAL_STRUCT_DICT[metal]]:
                surface = load_surface(metal=metal, hkl=facet)
                assert surface.num_atoms != 0
                assert len(surface.vacuum_height) >= 10.0

    def test_model(self):
        assert model_inter.model.parameters() != None

    def test_serial_eval(self):
        test_inters = random.sample(list(intermediates.values()), 3)
        for inter in test_inters:
            model_inter(inter)
            if inter.phase == "ads":
                assert len(inter.ads_configs) == 2
            elif inter.phase in ("gas", "surf"):
                assert len(inter.ads_configs) == 1
