import random
import unittest

from care import gen_blueprint
from care.evaluators import load_surface
from care.evaluators.ocp import OCPIntermediateEvaluator

intermediates, rxns = gen_blueprint(1, 1, False, False, False)
surface = load_surface("Fe", "110")
model_inter = OCPIntermediateEvaluator(surface, num_configs=2, max_steps=3)

class TestEvaluator(unittest.TestCase):
    def test_serial_eval(self):
        test_inters = random.sample(list(intermediates.values()), 3)
        for inter in test_inters:
            model_inter(inter)
            if inter.phase == "ads":
                assert len(inter.ads_configs) == 2
                self.assertAlmostEqual(inter.ads_configs["0"]["s"], 0.0, places=3)
                self.assertAlmostEqual(inter.ads_configs["1"]["s"], 0.0, places=3)
            elif inter.phase in ("gas", "surf"):
                assert len(inter.ads_configs) == 1
