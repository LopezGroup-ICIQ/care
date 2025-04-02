import unittest
import pytest

from tests import surface, test_inters


class TestEvaluator(unittest.TestCase):
    @pytest.mark.skip(reason="Failing only on GitHub Actions. Incompatible with e3nn package across evaluators.")
    def test_serial_eval(self):
        from care.evaluators.sevennet import SevenNetIntermediateEvaluator
        model_inter = SevenNetIntermediateEvaluator(surface,
                                            model="7net-mf-ompa",
                                            modal="mpa",
                                            num_configs=2,
                                            max_steps=2)
        for inter in test_inters:
            model_inter(inter)
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

    # As of 02-04-2025, SevenNet does not support parallel evaluation with ASE calculators.
    # def test_parallel_eval(self):
    #     ...
